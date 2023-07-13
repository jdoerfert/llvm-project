//===- ASTIR.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which simply prints the names of all the top-level decls
// in the input file.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

namespace {

struct Visitor : public RecursiveASTVisitor<Visitor> {
  LLVMContext Ctx;
  llvm::Module *M;
  clang::ASTContext &ASTCtx;

  /// Return whether this visitor should traverse post-order.
  bool shouldTraversePostOrder() const { return true; }

  llvm::IRBuilder<> Builder;
  llvm::BasicBlock *EntryBB;

  Visitor(clang::ASTContext &ASTCtx) : ASTCtx(ASTCtx), Builder(Ctx) {
    M = new llvm::Module("ASTIR", Ctx);
    EntryBB = llvm::BasicBlock::Create(Ctx);
  }
  ~Visitor() { M->dump(); }

  static std::string exprValueKindAsString(clang::ExprValueKind Kind) {
    switch (Kind) {
    case clang::VK_PRValue:
      return "pr";
    case clang::VK_LValue:
      return "l";
    case clang::VK_XValue:
      return "x";
    };
  }
  llvm::Type *convertType(clang::QualType QT, clang::ExprValueKind Kind) {
    return Builder.getPtrTy();
    llvm::Type *&Ty = TypeMap[{QT.getAsOpaquePtr(), int(Kind)}];
    if (!Ty)
      Ty = llvm::StructType::create(Ctx, QT.getAsString() + "." +
                                             exprValueKindAsString(Kind));
    return Ty;
  }

  llvm::Function *createNewFunction(Decl *D, clang::Stmt *S, StringRef Name,
                                    clang::QualType RetTy,
                                    ArrayRef<clang::ParmVarDecl *> Params) {
    SmallVector<llvm::Type *> ParamTypes;
    for (auto *Param : Params)
      ParamTypes.push_back(convertType(
          Param->getType(), Expr::getValueKindForType(Param->getType())));
    return createNewFunction(
        D, S, Name, convertType(RetTy, Expr::getValueKindForType(RetTy)),
        ParamTypes);
  }

  llvm::Function *createNewFunction(Decl *D, clang::Stmt *S, StringRef Name,
                                    llvm::Type *RetTy,
                                    ArrayRef<llvm::Type *> ParamTypes) {
    llvm::FunctionType *FTy = llvm::FunctionType::get(RetTy, ParamTypes, false);
    llvm::Function *F =
        Function::Create(FTy, llvm::GlobalValue::ExternalLinkage, 0, Name, M);
    return F;
  }

  bool VisitDecl(Decl *D) {
    errs() << "Visit\n";
    D->dump();
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    auto *Fn = createNewFunction(FD, nullptr, FD->getName(),
                                 FD->getReturnType(), FD->parameters());
    DeclMap[FD] = Fn;
    if (!FD->hasBody())
      return true;
    auto *AI = Fn->arg_begin();
    for (auto *Param : FD->parameters()) {
      auto *Old = cast<Instruction>(DeclMap[Param]);
      AI->setName(Old->getName());
      auto *Placeholder = Old->getOperand(0);
      Old->setOperand(0, AI++);
      Placeholder->deleteValue();
    }
    for (auto *I : StmtMap[FD->getBody()]) {
      I->insertInto(EntryBB, EntryBB->end());
    }
    llvm::ReturnInst::Create(Ctx, EntryBB);
    EntryBB->insertInto(Fn);
    EntryBB = llvm::BasicBlock::Create(Ctx);
    return true;
  }

  bool VisitParmVarDecl(ParmVarDecl *PAD) {
    if (!cast<FunctionDecl>(PAD->getDeclContext())->hasBody())
      return true;
    errs() << "Visit ParamVarDecl\n";
    PAD->dump();
    auto *A = new llvm::Argument(convertType(PAD->getType(), clang::VK_LValue),
                                 PAD->getName());
    auto *CS = DeclMap[PAD] = createCall(
        "param_var_decl", convertType(PAD->getType(), clang::VK_LValue),
        {A, Builder.CreateGlobalStringPtr(PAD->getName(), "var_name", 0, M)},
        false, false);
    CS->setName(PAD->getName());
    cast<Instruction>(CS)->insertInto(EntryBB, EntryBB->end());
    return true;
  }

  bool VisitDeclStmt(DeclStmt *DS) {
    errs() << "Visit DeclStmt\n";
    DS->dump();
    SmallVector<llvm::Value *, 4> Args;
    for (auto *D : DS->decls())
      Args.push_back(DeclMap[D]);
    createCall("decl_stmt", Builder.getVoidTy(), Args, true, true);
    return true;
  }
  bool VisitVarDecl(VarDecl *VD) {
    if (isa<ParmVarDecl>(VD))
      return true;
    SmallVector<llvm::Value *, 4> Args = {
        Builder.CreateGlobalStringPtr(VD->getName(), "var_name", 0, M)};
    if (auto *Init = VD->getInit())
      Args.push_back(ExprMap[Init]);
    auto *CS = DeclMap[VD] =
        createCall("var_decl", convertType(VD->getType(), clang::VK_LValue),
                   Args, false, true);
    CS->setName(VD->getName());
    return true;
  }

  bool VisitStmt(Stmt *S) {
    errs() << "Visit S:\n";
    S->dump();
    CurStmt = S;
    return true;
  }

  void addInst(llvm::Instruction &I, llvm::BasicBlock &BB,
               SmallVector<llvm::Value *> &Args,
               llvm::DenseMap<llvm::Value *, unsigned> &VMap) {
    for (auto *Op : I.operand_values()) {
      if (isa<Constant>(Op))
        continue;
      if (auto *OpI = dyn_cast<Instruction>(Op)) {
        if (OpI->getParent() == &BB)
          continue;
        if (!OpI->getParent()) {
          addInst(*OpI, BB, Args, VMap);
          continue;
        }
      }
      assert(isa<Argument>(Op) || isa<Instruction>(Op));
      unsigned &ArgNo = VMap[Op];
      if (ArgNo)
        continue;
      ArgNo = Args.size();
      Args.push_back(Op);
    }
    errs() << "Move " << I << " :: " << I.getNumUses() << "\n";
    I.insertInto(&BB, BB.end());
  }

  bool VisitCompoundStmt(CompoundStmt *CS) {
    auto *BoolTy = llvm::Type::getInt1Ty(Ctx);
    auto *VoidTy = llvm::Type::getVoidTy(Ctx);
    auto *PtrTy = llvm::PointerType::get(Ctx, 0);

    llvm::DenseMap<llvm::Value *, unsigned> VMap;
    SmallVector<llvm::Value *> Args;
    //    Args.push_back(ConstantPointerNull::get(PtrTy));

    BasicBlock *BB = BasicBlock::Create(Ctx, "cs");
    for (auto *C : CS->body()) {
      for (auto *I : StmtMap[C])
        addInst(*I, *BB, Args, VMap);
    }

    SmallVector<llvm::Type *> ArgTypes;
    for (auto *Arg : Args)
      ArgTypes.push_back(Arg->getType());

    Function *BodyFn =
        createNewFunction(nullptr, CS, "ast_ir.cs.body", VoidTy, ArgTypes);
    BB->insertInto(BodyFn);
    llvm::ReturnInst::Create(Ctx, BB);

    for (auto &It : VMap) {
      auto *Arg = BodyFn->getArg(It.getSecond());
      Arg->setName(It.getFirst()->getName());
      It.getFirst()->replaceUsesWithIf(Arg, [BodyFn](Use &U) {
        return llvm::isa_and_nonnull<Instruction>(U.getUser()) &&
               llvm::cast<Instruction>(U.getUser())->getParent() &&
               llvm::cast<Instruction>(U.getUser())->getFunction() == BodyFn;
      });
    }

    Args.insert(Args.begin(), BodyFn);
    createCall("compound_stmt", VoidTy, Args, true, true);
    return true;
  }

  bool VisitReturnStmt(clang::ReturnStmt *RS) {
    //    errs() << "Ret Stmt \n";
    //    RS->dump();
    //    Scp->DelayedStmtStack.push_back(
    //        DelayedStmt{"return_stmt", 1, Scp->Builder.getVoidTy()});
    auto *RV = RS->getRetValue();
    assert(!RV || ExprMap.count(RV));
    unsigned TS = RV ? ASTCtx.getTypeSize(RV->getType()) : 0;
    createCall("return_stmt",
               RV ? convertType(RV->getType(), RV->getValueKind())
                  : Builder.getVoidTy(),
               RV ? ArrayRef<llvm::Value *>({Builder.getInt32(TS), ExprMap[RV]})
                  : ArrayRef<llvm::Value *>(),
               true, true);
    return true;
  }

  bool VisitStringLiteral(clang::StringLiteral *SL) {
    errs() << "Viit SL: ";
    SL->dump();
    unsigned TS = ASTCtx.getTypeSize(SL->getType());
    ExprMap[SL] = createCall(
        "string_literal", convertType(SL->getType(), SL->getValueKind()),
        {Builder.getInt32(TS),
         Builder.CreateGlobalString(SL->getString(), "", 0, M)},
        false, true);
    return true;
  }

  bool VisitIntegerLiteral(IntegerLiteral *IL) {
    errs() << "Viit IL: ";
    IL->dump();
    unsigned TS = ASTCtx.getTypeSize(IL->getType());
    auto *Init = ConstantInt::get(Ctx, *IL->getIntegerConstantExpr(ASTCtx));
    ExprMap[IL] = createCall(
        "integer_literal", convertType(IL->getType(), IL->getValueKind()),
        {Builder.getInt32(TS),
         new llvm::GlobalVariable(*M, Init->getType(), true,
                                  llvm::GlobalVariable::InternalLinkage, Init)},
        false, true);
    return true;
  }

  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE) {
    assert(ExprMap.count(ASE->getLHS()));
    assert(ExprMap.count(ASE->getRHS()));
    unsigned TS = ASTCtx.getTypeSize(ASE->getType());
    ExprMap[ASE] = createCall(
        "array_subscript_expr",
        convertType(ASE->getType(), ASE->getValueKind()),
        {Builder.getInt32(TS), ExprMap[ASE->getLHS()], ExprMap[ASE->getRHS()]},
        false, true);
    return true;
  }

  bool VisitBinaryOperator(clang::BinaryOperator *BO) {
    assert(ExprMap.count(BO->getLHS()));
    assert(ExprMap.count(BO->getRHS()));
    unsigned TS = ASTCtx.getTypeSize(BO->getType());
    ExprMap[BO] = createCall(
        "binary_op", convertType(BO->getType(), BO->getValueKind()),
        {Builder.getInt32(TS), ExprMap[BO->getLHS()], ExprMap[BO->getRHS()]},
        false, true);
    return true;
  }

  bool VisitExpr(clang::Expr *E) { return true; }

  bool VisitCallExpr(clang::CallExpr *CE) {
    SmallVector<llvm::Value *, 4> Args;
    unsigned TS = ASTCtx.getTypeSize(CE->getType());
    Args.push_back(Builder.getInt32(TS));

    //    if (auto *CD = CE->getCalleeDecl())
    //      Args.push_back(DeclMap[CD]);
    //    else
    Args.push_back(
        Builder.getInt32(ASTCtx.getTypeSize(CE->getCallee()->getType())));
    Args.push_back(ExprMap[CE->getCallee()]);

    Args.push_back(Builder.getInt32(CE->getNumArgs()));
    for (auto *Arg : CE->arguments()) {
      assert(ExprMap.count(Arg));
      unsigned TS = ASTCtx.getTypeSize(Arg->getType());
      Args.push_back(Builder.getInt32(TS));
      Args.push_back(ExprMap[Arg]);
    }

    ExprMap[CE] =
        createCall("call_expr", convertType(CE->getType(), CE->getValueKind()),
                   Args, false, true);
    StmtMap[CE].push_back(cast<Instruction>(ExprMap[CE]));
    return true;
  }

  bool VisitImplicitCastExpr(clang::ImplicitCastExpr *ICE) {
    assert(ExprMap.count(ICE->getSubExpr()));
    unsigned TS = ASTCtx.getTypeSize(ICE->getType());
    ExprMap[ICE] = createCall("implicit_cast_expr",
                              convertType(ICE->getType(), ICE->getValueKind()),
                              {
                                  Builder.getInt32(TS),
                                  ExprMap[ICE->getSubExpr()],
                                  Builder.getInt32(ICE->getCastKind()),
                              },
                              false, true);
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    unsigned TS = ASTCtx.getTypeSize(DRE->getType());
    ExprMap[DRE] = createCall(
        "decl_ref", convertType(DRE->getType(), DRE->getValueKind()),
        {Builder.getInt32(TS), DeclMap[DRE->getDecl()]}, false, true);
    return true;
  }

  llvm::Instruction *createCall(StringRef Name, llvm::Type *RetTy,
                                ArrayRef<llvm::Value *> Args, bool IsStmt,
                                bool IsInsts) {
    SmallVector<llvm::Type *> ParamTypes;
    for (auto *V : Args) {
      assert(V);
      ParamTypes.push_back(V->getType());
    }

    llvm::FunctionType *FTy = llvm::FunctionType::get(RetTy, ParamTypes, false);
    FunctionCallee CSCallee =
        M->getOrInsertFunction(("ast_ir." + Name).str(), FTy);

    auto *Call = Builder.CreateCall(CSCallee, Args);
    if (IsInsts && IsStmt)
      StmtMap[CurStmt].push_back(Call);
    return Call;
  }

  DenseMap<std::pair<void *, int>, llvm::Type *> TypeMap;
  DenseMap<clang::Decl *, llvm::Value *> DeclMap;
  DenseMap<clang::Expr *, llvm::Value *> ExprMap;
  DenseMap<clang::Stmt *, SmallVector<llvm::Instruction *>> StmtMap;
  clang::Stmt *CurStmt = nullptr;
  llvm::Value *LastDeclStmt = nullptr;
};

class PrintFunctionsConsumer : public ASTConsumer {

public:
  PrintFunctionsConsumer() {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor V(Context);
    Context.getTranslationUnitDecl()->dump();
    V.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class PrintFunctionNamesAction : public PluginASTAction {

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<PrintFunctionsConsumer>();
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
    if (!Args.empty() && Args[0] == "help")
      PrintHelp(llvm::errs());

    return true;
  }
  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Help for ASTIR plugin goes here\n";
  }
};
}

static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
    X("ast-ir", "print function names");
