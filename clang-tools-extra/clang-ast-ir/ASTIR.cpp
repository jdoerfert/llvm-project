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

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

namespace {

struct Visitor;

struct Delayed {
  using FnTy = std::function<void()>;
  FnTy Fn;
  Delayed(FnTy &Fn) : Fn(Fn) {}
  Delayed(const Delayed &Other) : Fn(Other.Fn) {}
  ~Delayed() { Fn(); }
};

struct DelayedStmt {
  StringRef Name;
  unsigned NumOperands;
  llvm::Type *RetTy;
};

struct Scope {
  Scope(Visitor &V, clang::Decl *D, clang::Stmt *S, llvm::Function *F);
  ~Scope();

  llvm::Value *getClosure();

  bool nextStmt() {
    if (!S)
      return false;
    return ++SChildIt == S->child_end();
  }

  void addParamDecl(clang::ParmVarDecl *PVD);

  void addDecl(clang::Decl *D, llvm::Value *V);

  Visitor &V;
  clang::Decl *D = nullptr;
  clang::Stmt *S = nullptr;
  clang::Stmt::child_iterator SChildIt;
  llvm::Function *F;
  llvm::Value *Closure = nullptr;
  llvm::LLVMContext &Ctx;
  llvm::Function::arg_iterator ArgIt;
  llvm::IRBuilder<> Builder;
  DenseMap<clang::Decl *, std::pair<llvm::Value *, int>> DeclMap;
  SmallVector<DelayedStmt> DelayedStmtStack;
  std::deque<llvm::Value *> ExprStack;
};

struct Visitor : public RecursiveASTVisitor<Visitor> {
  LLVMContext Ctx;
  llvm::Module *M;

  /// Return whether this visitor should traverse post-order.
  bool shouldTraversePostOrder() const { return true; }

  SmallVector<Delayed *> DelayedStack;
  SmallVector<clang::Decl *> DeclStack;

  SmallVector<Scope *> ScopeStack;
  Scope *Scp = nullptr;

  llvm::IRBuilder<> Builder;

  Visitor() : Builder(Ctx) { M = new llvm::Module("ASTIR", Ctx); }
  ~Visitor() { M->dump(); }

  llvm::Type *convertType(clang::QualType QT) {
    llvm::Type *&Ty = TypeMap[QT.getAsOpaquePtr()];
    if (!Ty)
      Ty = llvm::StructType::create(Ctx, QT.getAsString());
    return Ty;
  }

  llvm::Function *createNewFunction(Decl *D, clang::Stmt *S, StringRef Name,
                                    clang::QualType RetTy,
                                    ArrayRef<clang::ParmVarDecl *> Params) {
    SmallVector<llvm::Type *> ParamTypes;
    for (auto *Param : Params)
      ParamTypes.push_back(convertType(Param->getType()));
    return createNewFunction(D, S, Name, convertType(RetTy), ParamTypes);
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
    //    if (DeclStack.size() > 1) {
    //      errs() << "Pop\n";
    //      DeclStack.back()->dump();
    //      Decl *LastDecl = DeclStack.pop_back_val();
    //      assert(isa<clang::FunctionDecl>(LastDecl));
    //      delete ScopeStack.pop_back_val();
    //      Scp = ScopeStack.back();
    //    }
    //    DeclStack.push_back(FD);
    //    errs() << "Visit FunctionDecl\n";
    //    //    VisitDecl(FD);
    auto *Fn = createNewFunction(FD, nullptr, FD->getName(),
                                 FD->getReturnType(), FD->parameters());
    auto *AI = Fn->arg_begin();
    for (auto *Param : FD->parameters()) {
      auto *Old = DeclMap[Param];
      Old->replaceAllUsesWith(AI++);
      Old->deleteValue();
    }
    BasicBlock *BB = BasicBlock::Create(Ctx, "", Fn);
    while (!Insts.empty())
      Insts.pop_back_val().first->insertInto(BB, BB->begin());
    llvm::ReturnInst::Create(Ctx, BB);
    return true;
  }

  bool VisitParmVarDecl(ParmVarDecl *PAD) {
    errs() << "Visit ParamVarDecl\n";
    PAD->dump();
    DeclMap[PAD] = createCall(PAD->getName(), convertType(PAD->getType()), {},
                              false, false);
    return true;
  }

  bool VisitStmt(Stmt *S) {
    errs() << "Visit S:\n";
    S->dump();
    //    if (Scp->nextStmt()) {
    //      delete ScopeStack.pop_back_val();
    //      Scp = ScopeStack.back();
    //    };
    return true;
  }

  bool VisitCompoundStmt(CompoundStmt *CS) {
    auto *BoolTy = llvm::Type::getInt1Ty(Ctx);
    auto *PtrTy = llvm::PointerType::get(Ctx, 0);
    //    FunctionCallee CSCallee =
    //        M->getOrInsertFunction("ast_ir.compound_stmt", BoolTy, PtrTy,
    //        PtrTy);
    //    SmallVector<llvm::Value *, 2> Args{ConstantPointerNull::get(PtrTy),
    //                                       Scp->getClosure()};
    //    CallBase *CB = Scp->Builder.CreateCall(CSCallee, Args);
    //    CB->setArgOperand(0, BodyFn);
    //
    //    BodyFn->arg_begin()->setName("ast_ir.closure");
    //    Scp->Closure = BodyFn->arg_begin();
    //
    //    if (CS->body_empty()) {
    //      delete ScopeStack.pop_back_val();
    //      Scp = ScopeStack.back();
    //    }

    Function *BodyFn =
        createNewFunction(nullptr, CS, "ast_ir.cs.body", BoolTy, {PtrTy});
    BasicBlock *BB = BasicBlock::Create(Ctx, "cs", BodyFn);
    for (unsigned I = CS->size(); I > 0;) {
      const auto &It = Insts.pop_back_val();
      It.first->insertInto(BB, BB->begin());
      I -= It.second;
    }
    llvm::ReturnInst::Create(Ctx, ConstantInt::getBool(Ctx, HitReturn), BB);
    createCall("compound_stmt", BoolTy, {BodyFn}, true, true);
    return true;
  }

  bool VisitReturnStmt(clang::ReturnStmt *RS) {
    //    errs() << "Ret Stmt \n";
    //    RS->dump();
    //    Scp->DelayedStmtStack.push_back(
    //        DelayedStmt{"return_stmt", 1, Scp->Builder.getVoidTy()});
    auto *RV = RS->getRetValue();
    assert(!RV || ExprMap.count(RV));
    createCall(
        "return_stmt", RV ? convertType(RV->getType()) : Builder.getVoidTy(),
        RV ? ArrayRef<llvm::Value *>(ExprMap[RV]) : ArrayRef<llvm::Value *>(),
        true, true);
    HitReturn = true;
    return true;
  }

  bool VisitBinaryOperator(clang::BinaryOperator *BO) {
    //    errs() << "Bin op \n";
    //    BO->dump();
    //    Scp->DelayedStmtStack.push_back(
    //        DelayedStmt{"binary_operator", 2, convertType(BO->getType())});
    assert(ExprMap.count(BO->getLHS()));
    assert(ExprMap.count(BO->getRHS()));
    ExprMap[BO] =
        createCall("binary_op", convertType(BO->getType()),
                   {ExprMap[BO->getLHS()], ExprMap[BO->getRHS()]}, false, true);
    return true;
  }
  bool VisitImplicitCastExpr(clang::ImplicitCastExpr *ICE) {
    assert(ExprMap.count(ICE->getSubExpr()));
    ExprMap[ICE] = createCall("implicit_cast_expr", convertType(ICE->getType()),
                              {ExprMap[ICE->getSubExpr()]}, false, true);
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    //    errs() << "Decl Ref\n";
    //    DRE->dump();
    //    Scp->DelayedStmtStack.push_back(
    //        DelayedStmt{"decl_ref_expr", 1, convertType(DRE->getType())});
    //    Scp->ExprStack.push_back(
    //        Scp->Builder.getInt32(Scp->DeclMap.lookup(DRE->getDecl()).second));
    ExprMap[DRE] = createCall("decl_ref", convertType(DRE->getType()),
                              {DeclMap[DRE->getDecl()]}, false, true);
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
    if (IsInsts)
      Insts.push_back({Call, IsStmt});
    return Call;
  }

  DenseMap<void *, llvm::Type *> TypeMap;
  DenseMap<clang::Decl *, llvm::Value *> DeclMap;
  DenseMap<clang::Expr *, llvm::Value *> ExprMap;
  SmallVector<std::pair<llvm::Instruction *, bool>> Insts;
  bool HitReturn = false;
};

class PrintFunctionsConsumer : public ASTConsumer {

public:
  PrintFunctionsConsumer() {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor V;
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
                 const std::vector<std::string> &args) override {
    if (!args.empty() && args[0] == "help")
      PrintHelp(llvm::errs());

    return true;
  }
  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Help for ASTIR plugin goes here\n";
  }
};

Scope::Scope(Visitor &V, clang::Decl *D, clang::Stmt *S, llvm::Function *F)
    : V(V), D(D), S(S), SChildIt(S ? S->child_begin() : decltype(SChildIt)()),
      F(F), Ctx(V.Ctx), ArgIt(F->arg_begin()), Builder(V.Ctx),
      DeclMap(!V.Scp ? decltype(DeclMap)() : V.Scp->DeclMap) {
  Builder.SetInsertPoint(llvm::BasicBlock::Create(V.Ctx, "", F));
};

Scope::~Scope() {
  errs() << "Delete scope : " << F->getName() << "\n";

  for (unsigned I = DelayedStmtStack.size(); I > 0; --I) {
    DelayedStmt &DS = DelayedStmtStack[I - 1];

    SmallVector<llvm::Value *, 4> Args;
    Args.resize(DS.NumOperands + 1);
    Args[0] = getClosure();
    for (unsigned PI = 0; PI < DS.NumOperands; ++PI) {
      Args[DS.NumOperands - PI] = ExprStack.back();
      ExprStack.pop_back();
    }

    SmallVector<llvm::Type *> ParamTypes;
    for (auto *V : Args)
      ParamTypes.push_back(V->getType());

    llvm::FunctionType *FTy =
        llvm::FunctionType::get(DS.RetTy, ParamTypes, false);
    FunctionCallee CSCallee =
        V.M->getOrInsertFunction(("ast_ir." + DS.Name).str(), FTy);

    CallBase *CB = Builder.CreateCall(CSCallee, Args);
    if (!DS.RetTy->isVoidTy())
      ExprStack.push_front(CB);
  }
}

llvm::Value *Scope::getClosure() { return Closure; }

void Scope::addParamDecl(clang::ParmVarDecl *PVD) {
  ArgIt->setName(PVD->getName());
  addDecl(PVD, ArgIt++);
}

void Scope::addDecl(clang::Decl *D, llvm::Value *Val) {
  unsigned Idx = DeclMap.size();
  DeclMap[D] = {Val, Idx};

  auto *PtrTy = Builder.getPtrTy();
  auto *I32Ty = Builder.getInt32Ty();

  SmallVector<llvm::Type *> ParamTypes{PtrTy, I32Ty, Val->getType()};

  llvm::FunctionType *FTy = llvm::FunctionType::get(PtrTy, ParamTypes, false);
  FunctionCallee CSCallee = V.M->getOrInsertFunction("ast_ir.add_decl", FTy);

  SmallVector<llvm::Value *, 4> Args{Closure, Builder.getInt32(Idx), Val};
  Closure = Builder.CreateCall(CSCallee, Args, "ast_ir.closure");
}

} // namespace

static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
    X("ast-ir", "print function names");
