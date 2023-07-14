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
#include "llvm/IR/Constant.h"
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
#include <system_error>

using namespace clang;
using namespace llvm;

namespace {

static std::string Prefix = "_CLAIR_";

struct Visitor : public RecursiveASTVisitor<Visitor> {
  clang::ASTContext &ASTCtx;

  LLVMContext Ctx;
  llvm::Module M;

  /// Return whether this visitor should traverse post-order.
  bool shouldTraversePostOrder() const { return true; }

  llvm::IRBuilder<> Builder;
  llvm::Function *DummyFn;
  llvm::BasicBlock *EntryBB;
  llvm::BasicBlock *CurBB;
  llvm::Type *BoxTy;

  Visitor(clang::ASTContext &ASTCtx)
      : ASTCtx(ASTCtx), M(llvm::Module("ASTIR", Ctx)), Builder(Ctx),
        BoxTy(llvm::PointerType::get(Ctx, 0)) {
    DummyFn = llvm::Function::Create(
        llvm::FunctionType::get(Builder.getVoidTy(), false),
        llvm::GlobalValue::ExternalLinkage, "", &M);
    EntryBB = llvm::BasicBlock::Create(Ctx, "", DummyFn);
    CurBB = llvm::BasicBlock::Create(Ctx, "", DummyFn);
    //		llvm::GlobalVariable(BoxTy, /* isConstant */ false,
    // llvm::GlobalValue::InternalLinkage, /* Initializer */ nullptr, Prefix +
    //"return_value",
  }
  ~Visitor() {
    M.dump();
    DummyFn->eraseFromParent();
    std::error_code EC;
    llvm::raw_fd_ostream OS("clair.ll", EC);
    M.print(OS, nullptr);
  }

  llvm::Function *createNewFunction(Twine Name,
                                    ArrayRef<clang::ParmVarDecl *> Params) {
    SmallVector<llvm::Type *> ParamTypes;
    for (auto *P : Params) {
      (void)P;
      ParamTypes.push_back(Builder.getPtrTy());
    }
    return createNewFunction(Name, ParamTypes);
  }

  llvm::Function *createNewFunction(Twine Name,
                                    ArrayRef<llvm::Type *> ParamTypes,
                                    llvm::GlobalValue::LinkageTypes Linkage =
                                        llvm::GlobalValue::ExternalLinkage) {
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(Builder.getVoidTy(), ParamTypes, false);
    llvm::Function *F = Function::Create(FTy, Linkage, 0, Name, &M);
    return F;
  }

  bool VisitDecl(Decl *D) {
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    auto *Fn =
        createNewFunction(Prefix + "_" + FD->getName(), FD->parameters());
    DeclMap[FD] = Fn;
    if (!FD->hasBody())
      return true;
    auto *AI = Fn->arg_begin();
    for (auto *Param : FD->parameters()) {
      auto *Old = cast<Instruction>(DeclMap[Param]);
      AI->takeName(Old);
      auto *Placeholder = Old->getOperand(0);
      Old->setOperand(0, AI++);
      Placeholder->deleteValue();
    }

    assert(CurBB->size() == 1);
    M.dump();
    CurBB->front().moveBefore(*EntryBB, EntryBB->end());

    llvm::ReturnInst::Create(Ctx, EntryBB);
    EntryBB->removeFromParent();
    EntryBB->insertInto(Fn);
    EntryBB = llvm::BasicBlock::Create(Ctx, "", DummyFn);
    return true;
  }

  bool VisitParmVarDecl(ParmVarDecl *PAD) {
    if (!cast<FunctionDecl>(PAD->getDeclContext())->hasBody())
      return true;
    auto *A = new llvm::Argument(Builder.getPtrTy(), PAD->getName());
    auto *CS = DeclMap[PAD] = createCall(
        "param_var_decl", Builder.getPtrTy(),
        {A, Builder.CreateGlobalStringPtr(PAD->getName(), "var_name", 0, &M)},
        false);
    cast<Instruction>(CS)->moveBefore(*EntryBB, EntryBB->end());
    CS->setName(PAD->getName() + "WWWW");
    return true;
  }

  bool VisitDeclStmt(DeclStmt *DS) {
    SmallVector<llvm::Value *, 4> Args;
    for (auto *D : DS->decls())
      Args.push_back(DeclMap[D]);
    createCall("decl_stmt", Builder.getVoidTy(), Args, true);
    return true;
  }
  bool VisitVarDecl(VarDecl *VD) {
    if (isa<ParmVarDecl>(VD))
      return true;
    SmallVector<llvm::Value *, 4> Args = {
        Builder.CreateGlobalStringPtr(VD->getName(), "var_name", 0, &M)};
    if (auto *Init = VD->getInit())
      Args.push_back(ExprMap[Init]);
    auto *CS = DeclMap[VD] =
        createCall("var_decl", Builder.getPtrTy(), Args, false);
    CS->setName(VD->getName());
    return true;
  }

  bool VisitStmt(Stmt *S) {
    CurStmt = S;
    return true;
  }

  bool VisitCompoundStmt(CompoundStmt *CS) {
    auto *VoidTy = llvm::Type::getVoidTy(Ctx);

    BasicBlock *BB = CurBB;
    CurBB = llvm::BasicBlock::Create(Ctx, "", DummyFn);

    llvm::SmallSetVector<llvm::Value *, 8> Args;
    llvm::SmallSetVector<llvm::Instruction *, 8> Worklist;
    for (auto *C : CS->body())
      Worklist.insert(StmtMap[C]);

    for (llvm::Instruction *I : Worklist) {
      for (llvm::Value *Op : I->operand_values()) {
        if (isa<Constant>(Op))
          continue;
        if (llvm::Instruction *OpI = dyn_cast<Instruction>(Op))
          if (Worklist.contains(OpI) || StmtInstSet.contains(OpI))
            continue;
        Args.insert(Op);
      }
    }

    SmallVector<llvm::Type *> StructTypes;
    for (auto *Arg : Args)
      StructTypes.push_back(Arg->getType());

    Builder.SetInsertPoint(EntryBB, EntryBB->getFirstInsertionPt());
    auto *ST = StructType::get(Ctx, StructTypes);
    auto *AL = Builder.CreateAlloca(ST);

    Builder.SetInsertPoint(EntryBB, EntryBB->end());
    for (unsigned I = 0; I < Args.size(); ++I) {
      llvm::Value *GEP = Builder.CreateStructGEP(ST, Args[I], I);
      Builder.CreateStore(Args[I], GEP);
    }

    Function *BodyFn =
        createNewFunction(Prefix + "cs.body", {Builder.getPtrTy()},
                          llvm::GlobalValue::PrivateLinkage);
    M.dump();
    BB->removeFromParent();
    BB->insertInto(BodyFn);
    llvm::ReturnInst::Create(Ctx, BB);

    Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
    for (unsigned I = 0; I < Args.size(); ++I) {
      llvm::Type *ElemTy = Args[I]->getType();
      llvm::Value *GEP = Builder.CreateStructGEP(ST, BodyFn->getArg(0), I);
      llvm::Value *Repl =
          Builder.CreateLoad(ElemTy, GEP, Args[I]->getName() + "QQQ");
      Args[I]->dump();
      //      Args[I]->replaceUsesWithIf(Repl, [this](Use &U) {
      //        return llvm::isa_and_nonnull<Instruction>(U.getUser()) &&
      //               llvm::cast<Instruction>(U.getUser())->getParent() !=
      //               EntryBB;
      //      });
    }

    createCall("compound_stmt", VoidTy, {BodyFn, AL}, true);
    return true;
  }

  bool VisitReturnStmt(clang::ReturnStmt *RS) {
    auto *RV = RS->getRetValue();
    assert(!RV || ExprMap.count(RV));
    unsigned TS = RV ? ASTCtx.getTypeSize(RV->getType()) : 0;
    createCall("return_stmt", RV ? Builder.getPtrTy() : Builder.getVoidTy(),
               RV ? ArrayRef<llvm::Value *>({Builder.getInt32(TS), ExprMap[RV]})
                  : ArrayRef<llvm::Value *>(),
               true);
    return true;
  }

  bool VisitStringLiteral(clang::StringLiteral *SL) {
    unsigned TS = ASTCtx.getTypeSize(SL->getType());
    ExprMap[SL] =
        createCall("string_literal", Builder.getPtrTy(),
                   {Builder.getInt32(TS),
                    Builder.CreateGlobalString(SL->getString(), "", 0, &M)},
                   false);
    return true;
  }

  bool VisitIntegerLiteral(IntegerLiteral *IL) {
    unsigned TS = ASTCtx.getTypeSize(IL->getType());
    auto *Init = ConstantInt::get(Ctx, *IL->getIntegerConstantExpr(ASTCtx));
    ExprMap[IL] = createCall(
        "integer_literal", Builder.getPtrTy(),
        {Builder.getInt32(TS),
         new llvm::GlobalVariable(M, Init->getType(), true,
                                  llvm::GlobalVariable::InternalLinkage, Init)},
        false);
    return true;
  }

  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE) {
    assert(ExprMap.count(ASE->getLHS()));
    assert(ExprMap.count(ASE->getRHS()));
    unsigned TS = ASTCtx.getTypeSize(ASE->getType());
    ExprMap[ASE] = createCall(
        "array_subscript_expr", Builder.getPtrTy(),
        {Builder.getInt32(TS), ExprMap[ASE->getLHS()], ExprMap[ASE->getRHS()]},
        false);
    return true;
  }

  bool VisitBinaryOperator(clang::BinaryOperator *BO) {
    assert(ExprMap.count(BO->getLHS()));
    assert(ExprMap.count(BO->getRHS()));
    unsigned TS = ASTCtx.getTypeSize(BO->getType());
    ExprMap[BO] = createCall(
        "binary_op", Builder.getPtrTy(),
        {Builder.getInt32(TS), ExprMap[BO->getLHS()], ExprMap[BO->getRHS()]},
        false);
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

    ExprMap[CE] = createCall("call_expr", Builder.getPtrTy(), Args, true);
    return true;
  }

  bool VisitImplicitCastExpr(clang::ImplicitCastExpr *ICE) {
    assert(ExprMap.count(ICE->getSubExpr()));
    unsigned TS = ASTCtx.getTypeSize(ICE->getType());
    ExprMap[ICE] = createCall("implicit_cast_expr", Builder.getPtrTy(),
                              {
                                  Builder.getInt32(TS),
                                  ExprMap[ICE->getSubExpr()],
                                  Builder.getInt32(ICE->getCastKind()),
                              },
                              false);
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    unsigned TS = ASTCtx.getTypeSize(DRE->getType());
    errs() << "DRE:" << *DeclMap[DRE->getDecl()] << "\n";
    ExprMap[DRE] =
        createCall("decl_ref", Builder.getPtrTy(),
                   {Builder.getInt32(TS), DeclMap[DRE->getDecl()]}, false);
    return true;
  }

  llvm::Instruction *createCall(StringRef Name, llvm::Type *RetTy,
                                ArrayRef<llvm::Value *> Args, bool IsStmt) {
    SmallVector<llvm::Type *> ParamTypes;
    for (auto *V : Args) {
      assert(V);
      ParamTypes.push_back(V->getType());
    }

    llvm::FunctionType *FTy = llvm::FunctionType::get(RetTy, ParamTypes, false);
    FunctionCallee CSCallee = M.getOrInsertFunction((Prefix + Name).str(), FTy);

    Builder.SetInsertPoint(CurBB, CurBB->end());
    auto *Call = Builder.CreateCall(CSCallee, Args);
    if (IsStmt) {
      StmtMap[CurStmt] = Call;
      StmtInstSet.insert(Call);
    }
    return Call;
  }

  DenseMap<std::pair<void *, int>, llvm::Type *> TypeMap;
  DenseMap<clang::Decl *, llvm::Value *> DeclMap;
  DenseMap<clang::Expr *, llvm::Value *> ExprMap;
  DenseMap<clang::Stmt *, llvm::Instruction *> StmtMap;
  DenseSet<llvm::Instruction *> StmtInstSet;
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
      printHelp(llvm::errs());

    return true;
  }
  void printHelp(llvm::raw_ostream &OS) {
    OS << "Help for ASTIR plugin goes here\n";
  }
};
} // namespace

static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
    X("ast-ir", "print function names");
