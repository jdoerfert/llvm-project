; ModuleID = '/data/benchmarks/llvm-test-suite/MultiSource/Benchmarks/Ptrdist/anagram/anagram.c'
source_filename = "/data/benchmarks/llvm-test-suite/MultiSource/Benchmarks/Ptrdist/anagram/anagram.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%struct.Letter = type { i32, i32, i32, i32 }
%struct.Word = type { [2 x i64], i8*, i32 }
%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }
%struct.stat = type { i64, i64, i64, i32, i32, i32, i32, i64, i64, i64, i64, %struct.timespec, %struct.timespec, %struct.timespec, [3 x i64] }
%struct.timespec = type { i64, i64 }

@cchMinLength = dso_local local_unnamed_addr global i32 3, align 4
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [24 x i8] c"Cannot stat dictionary\0A\00", align 1
@pchDictionary = dso_local local_unnamed_addr global i8* null, align 8
@.str.1 = private unnamed_addr constant [42 x i8] c"Unable to allocate memory for dictionary\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.3 = private unnamed_addr constant [24 x i8] c"Cannot open dictionary\0A\00", align 1
@.str.4 = private unnamed_addr constant [32 x i8] c"main dictionary has %u entries\0A\00", align 1
@.str.5 = private unnamed_addr constant [41 x i8] c"Dictionary too large; increase MAXWORDS\0A\00", align 1
@.str.6 = private unnamed_addr constant [18 x i8] c"%lu bytes wasted\0A\00", align 1
@alPhrase = dso_local local_unnamed_addr global [26 x %struct.Letter] zeroinitializer, align 16
@aqMainMask = dso_local global [2 x i64] zeroinitializer, align 16
@aqMainSign = dso_local local_unnamed_addr global [2 x i64] zeroinitializer, align 16
@cchPhraseLength = dso_local local_unnamed_addr global i32 0, align 4
@auGlobalFrequency = dso_local local_unnamed_addr global [26 x i32] zeroinitializer, align 32
@.str.7 = private unnamed_addr constant [28 x i8] c"MAX_QUADS not large enough\0A\00", align 1
@.str.8 = private unnamed_addr constant [35 x i8] c"Out of memory after %d candidates\0A\00", align 1
@cpwCand = dso_local local_unnamed_addr global i32 0, align 4
@.str.9 = private unnamed_addr constant [4 x i8] c"%s \00", align 1
@.str.10 = private unnamed_addr constant [21 x i8] c"Too many candidates\0A\00", align 1
@apwCand = dso_local global [5000 x %struct.Word*] zeroinitializer, align 16
@.str.11 = private unnamed_addr constant [15 x i8] c"%d candidates\0A\00", align 1
@.str.12 = private unnamed_addr constant [7 x i8] c"%15s%c\00", align 1
@DumpWords.X = internal unnamed_addr global i32 0, align 4
@cpwLast = dso_local local_unnamed_addr global i32 0, align 4
@apwSol = dso_local local_unnamed_addr global [51 x %struct.Word*] zeroinitializer, align 16
@achByFrequency = dso_local global [26 x i8] zeroinitializer, align 16
@.str.14 = private unnamed_addr constant [25 x i8] c"Order of search will be \00", align 1
@fInteractive = dso_local local_unnamed_addr global i32 0, align 4
@stdout = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@stdin = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.16 = private unnamed_addr constant [36 x i8] c"Usage: anagram dictionary [length]\0A\00", align 1
@achPhrase = dso_local global [255 x i8] zeroinitializer, align 16
@.str.17 = private unnamed_addr constant [16 x i8] c"New length: %d\0A\00", align 1
@jbAnagram = dso_local global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

; Function Attrs: noinline nounwind uwtable
define dso_local void @Fatal(i8* nocapture readonly %pchMsg, i32 %u) local_unnamed_addr #0 {
entry:
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* %pchMsg, i32 %u) #13
  tail call void @exit(i32 1) #14
  unreachable
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @fprintf(%struct._IO_FILE* nocapture noundef, i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define dso_local void @ReadDict(i8* %pchFile) local_unnamed_addr #3 {
entry:
  %statBuf = alloca %struct.stat, align 8
  %0 = bitcast %struct.stat* %statBuf to i8*
  call void @llvm.lifetime.start.p0i8(i64 144, i8* nonnull %0) #15
  %call.i = call i32 @__xstat(i32 1, i8* nonnull %pchFile, %struct.stat* nonnull %statBuf) #15
  %tobool.not = icmp eq i32 %call.i, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @Fatal(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str, i64 0, i64 0), i32 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %st_size = getelementptr inbounds %struct.stat, %struct.stat* %statBuf, i64 0, i32 8
  %1 = load i64, i64* %st_size, align 8, !tbaa !6
  %add = add i64 %1, 52000
  %call1 = call noalias i8* @malloc(i64 %add) #15
  store i8* %call1, i8** @pchDictionary, align 8, !tbaa !2
  %cmp = icmp eq i8* %call1, null
  br i1 %cmp, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.end
  call void @Fatal(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.1, i64 0, i64 0), i32 0)
  br label %if.end3

if.end3:                                          ; preds = %if.then2, %if.end
  %call4 = call %struct._IO_FILE* @fopen(i8* nonnull %pchFile, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0))
  %cmp5 = icmp eq %struct._IO_FILE* %call4, null
  br i1 %cmp5, label %if.then6, label %if.end7

if.then6:                                         ; preds = %if.end3
  call void @Fatal(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.3, i64 0, i64 0), i32 0)
  br label %if.end7

if.end7:                                          ; preds = %if.then6, %if.end3
  %call861 = call i32 @feof(%struct._IO_FILE* %call4) #15
  %tobool9.not62 = icmp eq i32 %call861, 0
  br i1 %tobool9.not62, label %while.body, label %while.end25

while.body:                                       ; preds = %if.end7, %while.end
  %cWords.064 = phi i32 [ %inc24, %while.end ], [ 0, %if.end7 ]
  %pchBase.063 = phi i8* [ %incdec.ptr20, %while.end ], [ %call1, %if.end7 ]
  %add.ptr = getelementptr inbounds i8, i8* %pchBase.063, i64 2
  br label %while.cond10

while.cond10:                                     ; preds = %while.body14, %while.body
  %cLetters.0 = phi i32 [ 0, %while.body ], [ %spec.select, %while.body14 ]
  %pch.0 = phi i8* [ %add.ptr, %while.body ], [ %incdec.ptr, %while.body14 ]
  %call11 = call i32 @fgetc(%struct._IO_FILE* %call4)
  switch i32 %call11, label %while.body14 [
    i32 -1, label %while.end
    i32 10, label %while.end
  ]

while.body14:                                     ; preds = %while.cond10
  %call15 = tail call i16** @__ctype_b_loc() #16
  %2 = load i16*, i16** %call15, align 8, !tbaa !2
  %idxprom = sext i32 %call11 to i64
  %arrayidx = getelementptr inbounds i16, i16* %2, i64 %idxprom
  %3 = load i16, i16* %arrayidx, align 2, !tbaa !11
  %4 = lshr i16 %3, 10
  %.lobit = and i16 %4, 1
  %5 = zext i16 %.lobit to i32
  %spec.select = add i32 %cLetters.0, %5
  %conv19 = trunc i32 %call11 to i8
  %incdec.ptr = getelementptr inbounds i8, i8* %pch.0, i64 1
  store i8 %conv19, i8* %pch.0, align 1, !tbaa !13
  br label %while.cond10

while.end:                                        ; preds = %while.cond10, %while.cond10
  %incdec.ptr20 = getelementptr inbounds i8, i8* %pch.0, i64 1
  store i8 0, i8* %pch.0, align 1, !tbaa !13
  %sub.ptr.lhs.cast = ptrtoint i8* %incdec.ptr20 to i64
  %sub.ptr.rhs.cast = ptrtoint i8* %pchBase.063 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %conv21 = trunc i64 %sub.ptr.sub to i8
  store i8 %conv21, i8* %pchBase.063, align 1, !tbaa !13
  %conv22 = trunc i32 %cLetters.0 to i8
  %arrayidx23 = getelementptr inbounds i8, i8* %pchBase.063, i64 1
  store i8 %conv22, i8* %arrayidx23, align 1, !tbaa !13
  %inc24 = add i32 %cWords.064, 1
  %call8 = call i32 @feof(%struct._IO_FILE* %call4) #15
  %tobool9.not = icmp eq i32 %call8, 0
  br i1 %tobool9.not, label %while.body, label %while.end25

while.end25:                                      ; preds = %while.end, %if.end7
  %pchBase.0.lcssa = phi i8* [ %call1, %if.end7 ], [ %incdec.ptr20, %while.end ]
  %cWords.0.lcssa = phi i32 [ 0, %if.end7 ], [ %inc24, %while.end ]
  %call26 = call i32 @fclose(%struct._IO_FILE* %call4)
  store i8 0, i8* %pchBase.0.lcssa, align 1, !tbaa !13
  %6 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call28 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %6, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.4, i64 0, i64 0), i32 %cWords.0.lcssa) #13
  %cmp29 = icmp ugt i32 %cWords.0.lcssa, 25999
  br i1 %cmp29, label %if.then31, label %if.end32

if.then31:                                        ; preds = %while.end25
  call void @Fatal(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @.str.5, i64 0, i64 0), i32 0)
  br label %if.end32

if.end32:                                         ; preds = %if.then31, %while.end25
  %incdec.ptr27 = getelementptr inbounds i8, i8* %pchBase.0.lcssa, i64 1
  %7 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %8 = load i64, i64* bitcast (i8** @pchDictionary to i64*), align 8, !tbaa !2
  %sub.ptr.lhs.cast33 = ptrtoint i8* %incdec.ptr27 to i64
  %sub.ptr.sub35.neg = sub i64 %add, %sub.ptr.lhs.cast33
  %sub = add i64 %sub.ptr.sub35.neg, %8
  %call36 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %7, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.6, i64 0, i64 0), i64 %sub) #13
  call void @llvm.lifetime.end.p0i8(i64 144, i8* nonnull %0) #15
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noalias noundef %struct._IO_FILE* @fopen(i8* nocapture noundef readonly, i8* nocapture noundef readonly) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @feof(%struct._IO_FILE* nocapture noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @fgetc(%struct._IO_FILE* nocapture noundef) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare dso_local i16** @__ctype_b_loc() local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @fclose(%struct._IO_FILE* nocapture noundef) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: nounwind uwtable
define dso_local void @BuildMask(i8* nocapture readonly %pchPhrase) local_unnamed_addr #3 {
entry:
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(416) bitcast ([26 x %struct.Letter]* @alPhrase to i8*), i8 0, i64 416, i1 false)
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(16) bitcast ([2 x i64]* @aqMainMask to i8*), i8 0, i64 16, i1 false)
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(16) bitcast ([2 x i64]* @aqMainSign to i8*), i8 0, i64 16, i1 false)
  store i32 0, i32* @cchPhraseLength, align 4, !tbaa !14
  %0 = load i8, i8* %pchPhrase, align 1, !tbaa !13
  %cmp.not117 = icmp eq i8 %0, 0
  br i1 %cmp.not117, label %for.body.preheader, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %call = tail call i16** @__ctype_b_loc() #16
  %1 = load i16*, i16** %call, align 8, !tbaa !2
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %if.end15
  %2 = phi i32 [ 0, %while.body.lr.ph ], [ %7, %if.end15 ]
  %3 = phi i8 [ %0, %while.body.lr.ph ], [ %8, %if.end15 ]
  %pchPhrase.pn = phi i8* [ %pchPhrase, %while.body.lr.ph ], [ %incdec.ptr118, %if.end15 ]
  %incdec.ptr118 = getelementptr inbounds i8, i8* %pchPhrase.pn, i64 1
  %idxprom = sext i8 %3 to i64
  %arrayidx = getelementptr inbounds i16, i16* %1, i64 %idxprom
  %4 = load i16, i16* %arrayidx, align 2, !tbaa !11
  %5 = and i16 %4, 1024
  %tobool.not = icmp eq i16 %5, 0
  br i1 %tobool.not, label %if.end15, label %if.end

if.end:                                           ; preds = %while.body
  %call.i = tail call i32** @__ctype_tolower_loc() #16
  %.pn = load i32*, i32** %call.i, align 8, !tbaa !2
  %__res.0.in = getelementptr inbounds i32, i32* %.pn, i64 %idxprom
  %__res.0 = load i32, i32* %__res.0.in, align 4, !tbaa !14
  %sub = add nsw i32 %__res.0, -97
  %idxprom12 = sext i32 %sub to i64
  %uFrequency = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %idxprom12, i32 0
  %6 = load i32, i32* %uFrequency, align 16, !tbaa !15
  %inc = add i32 %6, 1
  store i32 %inc, i32* %uFrequency, align 16, !tbaa !15
  %inc14 = add nsw i32 %2, 1
  store i32 %inc14, i32* @cchPhraseLength, align 4, !tbaa !14
  br label %if.end15

if.end15:                                         ; preds = %if.end, %while.body
  %7 = phi i32 [ %inc14, %if.end ], [ %2, %while.body ]
  %8 = load i8, i8* %incdec.ptr118, align 1, !tbaa !13
  %cmp.not = icmp eq i8 %8, 0
  br i1 %cmp.not, label %for.body.preheader, label %while.body

for.body.preheader:                               ; preds = %if.end15, %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.inc74
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc74 ], [ 0, %for.body.preheader ]
  %cbtUsed.0125 = phi i32 [ %cbtUsed.2, %for.inc74 ], [ 0, %for.body.preheader ]
  %iq.0124 = phi i32 [ %iq.2, %for.inc74 ], [ 0, %for.body.preheader ]
  %uFrequency20 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 0
  %9 = load i32, i32* %uFrequency20, align 16, !tbaa !15
  %cmp21 = icmp eq i32 %9, 0
  %arrayidx25 = getelementptr inbounds [26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 %indvars.iv
  br i1 %cmp21, label %if.then23, label %for.inc.preheader

if.then23:                                        ; preds = %for.body
  store i32 -1, i32* %arrayidx25, align 4, !tbaa !14
  br label %for.inc74

for.inc.preheader:                                ; preds = %for.body
  store i32 0, i32* %arrayidx25, align 4, !tbaa !14
  %conv33 = zext i32 %9 to i64
  %inc37.1 = add nuw nsw i32 1, 1
  br label %for.inc

for.inc:                                          ; preds = %for.inc.for.inc_crit_edge, %for.inc.preheader
  %qNeed.0122 = phi i64 [ %shl, %for.inc.for.inc_crit_edge ], [ 1, %for.inc.preheader ]
  %inc37.phi = phi i32 [ %inc37.0, %for.inc.for.inc_crit_edge ], [ %inc37.1, %for.inc.preheader ]
  %shl = shl i64 %qNeed.0122, 1
  %cmp34.not = icmp ugt i64 %shl, %conv33
  br i1 %cmp34.not, label %for.end, label %for.inc.for.inc_crit_edge

for.inc.for.inc_crit_edge:                        ; preds = %for.inc
  %inc37.0 = add nuw nsw i32 %inc37.phi, 1
  br label %for.inc

for.end:                                          ; preds = %for.inc
  %add = add nsw i32 %inc37.phi, %cbtUsed.0125
  %cmp39 = icmp ugt i32 %add, 64
  br i1 %cmp39, label %if.then41, label %if.end47

if.then41:                                        ; preds = %for.end
  %inc42 = add i32 %iq.0124, 1
  %cmp43 = icmp ugt i32 %inc42, 1
  br i1 %cmp43, label %if.then45, label %if.end47

if.then45:                                        ; preds = %if.then41
  tail call void @Fatal(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.7, i64 0, i64 0), i32 0)
  %.pre = load i32, i32* %uFrequency20, align 16, !tbaa !15
  %.pre127 = zext i32 %.pre to i64
  br label %if.end47

if.end47:                                         ; preds = %if.then41, %if.then45, %for.end
  %conv61.pre-phi = phi i64 [ %conv33, %if.then41 ], [ %.pre127, %if.then45 ], [ %conv33, %for.end ]
  %iq.1 = phi i32 [ %inc42, %if.then41 ], [ %inc42, %if.then45 ], [ %iq.0124, %for.end ]
  %cbtUsed.1 = phi i32 [ 0, %if.then41 ], [ 0, %if.then45 ], [ %cbtUsed.0125, %for.end ]
  %10 = trunc i64 %shl to i32
  %conv49 = add i32 %10, -1
  %uBits = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 2
  store i32 %conv49, i32* %uBits, align 8, !tbaa !17
  %sh_prom = zext i32 %cbtUsed.1 to i64
  %spec.select = shl i64 %shl, %sh_prom
  %idxprom56 = zext i32 %iq.1 to i64
  %arrayidx57 = getelementptr inbounds [2 x i64], [2 x i64]* @aqMainSign, i64 0, i64 %idxprom56
  %11 = load i64, i64* %arrayidx57, align 8, !tbaa !18
  %or = or i64 %11, %spec.select
  store i64 %or, i64* %arrayidx57, align 8, !tbaa !18
  %shl63 = shl i64 %conv61.pre-phi, %sh_prom
  %arrayidx65 = getelementptr inbounds [2 x i64], [2 x i64]* @aqMainMask, i64 0, i64 %idxprom56
  %12 = load i64, i64* %arrayidx65, align 8, !tbaa !18
  %or66 = or i64 %shl63, %12
  store i64 %or66, i64* %arrayidx65, align 8, !tbaa !18
  %uShift = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 1
  store i32 %cbtUsed.1, i32* %uShift, align 4, !tbaa !19
  %iq71 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 3
  store i32 %iq.1, i32* %iq71, align 4, !tbaa !20
  %add72 = add nsw i32 %cbtUsed.1, %inc37.phi
  br label %for.inc74

for.inc74:                                        ; preds = %if.then23, %if.end47
  %iq.2 = phi i32 [ %iq.0124, %if.then23 ], [ %iq.1, %if.end47 ]
  %cbtUsed.2 = phi i32 [ %cbtUsed.0125, %if.then23 ], [ %add72, %if.end47 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 26
  br i1 %exitcond.not, label %for.end76, label %for.body

for.end76:                                        ; preds = %for.inc74
  ret void
}

; Function Attrs: argmemonly nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #6

; Function Attrs: nounwind readnone
declare dso_local i32** @__ctype_tolower_loc() local_unnamed_addr #5

; Function Attrs: nounwind uwtable
define dso_local noalias %struct.Word* @NewWord() local_unnamed_addr #3 {
entry:
  %call = tail call noalias dereferenceable_or_null(32) i8* @malloc(i64 32) #15
  %cmp = icmp eq i8* %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load i32, i32* @cpwCand, align 4, !tbaa !14
  tail call void @Fatal(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.8, i64 0, i64 0), i32 %0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = bitcast i8* %call to %struct.Word*
  ret %struct.Word* %1
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @wprint(i8* %pch) local_unnamed_addr #7 {
entry:
  %call = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), i8* %pch)
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local %struct.Word* @NextWord() local_unnamed_addr #3 {
entry:
  %0 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %cmp = icmp ugt i32 %0, 4999
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @Fatal(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.10, i64 0, i64 0), i32 0)
  %.pre = load i32, i32* @cpwCand, align 4, !tbaa !14
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = phi i32 [ %.pre, %if.then ], [ %0, %entry ]
  %inc = add i32 %1, 1
  store i32 %inc, i32* @cpwCand, align 4, !tbaa !14
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %idxprom
  %2 = load %struct.Word*, %struct.Word** %arrayidx, align 8, !tbaa !2
  %cmp1.not = icmp eq %struct.Word* %2, null
  br i1 %cmp1.not, label %if.end3, label %cleanup

if.end3:                                          ; preds = %if.end
  %call.i = tail call noalias dereferenceable_or_null(32) i8* @malloc(i64 32) #15
  %cmp.i = icmp eq i8* %call.i, null
  br i1 %cmp.i, label %if.then.i, label %NewWord.exit

if.then.i:                                        ; preds = %if.end3
  tail call void @Fatal(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.8, i64 0, i64 0), i32 %inc) #15
  %.pre11 = load i32, i32* @cpwCand, align 4, !tbaa !14
  br label %NewWord.exit

NewWord.exit:                                     ; preds = %if.end3, %if.then.i
  %3 = phi i32 [ %inc, %if.end3 ], [ %.pre11, %if.then.i ]
  %4 = bitcast i8* %call.i to %struct.Word*
  %sub = add i32 %3, -1
  %idxprom4 = zext i32 %sub to i64
  %arrayidx5 = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %idxprom4
  %5 = bitcast %struct.Word** %arrayidx5 to i8**
  store i8* %call.i, i8** %5, align 8, !tbaa !2
  br label %cleanup

cleanup:                                          ; preds = %if.end, %NewWord.exit
  %retval.0 = phi %struct.Word* [ %4, %NewWord.exit ], [ %2, %if.end ]
  ret %struct.Word* %retval.0
}

; Function Attrs: nounwind uwtable
define dso_local void @BuildWord(i8* %pchWord) local_unnamed_addr #3 {
entry:
  %cchFrequency = alloca [26 x i8], align 16
  %0 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 26, i8* nonnull %0) #15
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(26) %0, i8 0, i64 26, i1 false)
  %1 = load i8, i8* %pchWord, align 1, !tbaa !13
  %cmp.not90 = icmp eq i8 %1, 0
  br i1 %cmp.not90, label %for.cond.preheader, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  %call = tail call i16** @__ctype_b_loc() #16
  %2 = load i16*, i16** %call, align 8, !tbaa !2
  br label %while.body

for.cond.preheader.loopexit:                      ; preds = %while.cond.backedge
  %3 = bitcast [26 x i8]* %cchFrequency to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3, align 16, !tbaa !13
  %arrayidx26.8.phi.trans.insert = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 8
  %5 = bitcast i8* %arrayidx26.8.phi.trans.insert to <8 x i8>*
  %6 = load <8 x i8>, <8 x i8>* %5, align 8, !tbaa !13
  %arrayidx26.16.phi.trans.insert = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 16
  %.pre112 = load i8, i8* %arrayidx26.16.phi.trans.insert, align 16, !tbaa !13
  %7 = zext <8 x i8> %4 to <8 x i32>
  %8 = zext <8 x i8> %6 to <8 x i32>
  %phi.cast128 = zext i8 %.pre112 to i32
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %for.cond.preheader.loopexit, %entry
  %9 = phi i32 [ 0, %entry ], [ %phi.cast128, %for.cond.preheader.loopexit ]
  %cchLength.0.lcssa = phi i32 [ 0, %entry ], [ %cchLength.0.be, %for.cond.preheader.loopexit ]
  %10 = phi <8 x i32> [ zeroinitializer, %entry ], [ %7, %for.cond.preheader.loopexit ]
  %11 = phi <8 x i32> [ zeroinitializer, %entry ], [ %8, %for.cond.preheader.loopexit ]
  %arrayidx26.1 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 1
  %arrayidx26.2 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 2
  %arrayidx26.3 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 3
  %arrayidx26.4 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 4
  %arrayidx26.5 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 5
  %arrayidx26.6 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 6
  %arrayidx26.7 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 7
  %12 = load <8 x i32>, <8 x i32>* bitcast ([26 x i32]* @auGlobalFrequency to <8 x i32>*), align 32, !tbaa !14
  %13 = add <8 x i32> %12, %10
  store <8 x i32> %13, <8 x i32>* bitcast ([26 x i32]* @auGlobalFrequency to <8 x i32>*), align 32, !tbaa !14
  %arrayidx26.8 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 8
  %arrayidx26.9 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 9
  %arrayidx26.10 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 10
  %arrayidx26.11 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 11
  %arrayidx26.12 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 12
  %arrayidx26.13 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 13
  %arrayidx26.14 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 14
  %arrayidx26.15 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 15
  %14 = load <8 x i32>, <8 x i32>* bitcast (i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 8) to <8 x i32>*), align 32, !tbaa !14
  %15 = add <8 x i32> %14, %11
  store <8 x i32> %15, <8 x i32>* bitcast (i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 8) to <8 x i32>*), align 32, !tbaa !14
  %arrayidx26.16 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 16
  %arrayidx26.17 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 17
  %16 = load i8, i8* %arrayidx26.17, align 1, !tbaa !13
  %conv27.17 = zext i8 %16 to i32
  %arrayidx26.18 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 18
  %17 = load i8, i8* %arrayidx26.18, align 2, !tbaa !13
  %conv27.18 = zext i8 %17 to i32
  %arrayidx26.19 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 19
  %18 = load i8, i8* %arrayidx26.19, align 1, !tbaa !13
  %conv27.19 = zext i8 %18 to i32
  %arrayidx26.20 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 20
  %19 = load i8, i8* %arrayidx26.20, align 4, !tbaa !13
  %conv27.20 = zext i8 %19 to i32
  %arrayidx26.21 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 21
  %20 = load i8, i8* %arrayidx26.21, align 1, !tbaa !13
  %conv27.21 = zext i8 %20 to i32
  %arrayidx26.22 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 22
  %21 = load i8, i8* %arrayidx26.22, align 2, !tbaa !13
  %conv27.22 = zext i8 %21 to i32
  %arrayidx26.23 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 23
  %22 = load i8, i8* %arrayidx26.23, align 1, !tbaa !13
  %conv27.23 = zext i8 %22 to i32
  %23 = load <8 x i32>, <8 x i32>* bitcast (i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 16) to <8 x i32>*), align 32, !tbaa !14
  %24 = insertelement <8 x i32> undef, i32 %9, i32 0
  %25 = insertelement <8 x i32> %24, i32 %conv27.17, i32 1
  %26 = insertelement <8 x i32> %25, i32 %conv27.18, i32 2
  %27 = insertelement <8 x i32> %26, i32 %conv27.19, i32 3
  %28 = insertelement <8 x i32> %27, i32 %conv27.20, i32 4
  %29 = insertelement <8 x i32> %28, i32 %conv27.21, i32 5
  %30 = insertelement <8 x i32> %29, i32 %conv27.22, i32 6
  %31 = insertelement <8 x i32> %30, i32 %conv27.23, i32 7
  %32 = add <8 x i32> %23, %31
  store <8 x i32> %32, <8 x i32>* bitcast (i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 16) to <8 x i32>*), align 32, !tbaa !14
  %arrayidx26.24 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 24
  %33 = load i8, i8* %arrayidx26.24, align 8, !tbaa !13
  %conv27.24 = zext i8 %33 to i32
  %34 = load i32, i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 24), align 32, !tbaa !14
  %add.24 = add i32 %34, %conv27.24
  store i32 %add.24, i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 24), align 32, !tbaa !14
  %arrayidx26.25 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 25
  %35 = load i8, i8* %arrayidx26.25, align 1, !tbaa !13
  %conv27.25 = zext i8 %35 to i32
  %36 = load i32, i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 25), align 4, !tbaa !14
  %add.25 = add i32 %36, %conv27.25
  store i32 %add.25, i32* getelementptr inbounds ([26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 25), align 4, !tbaa !14
  %37 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %cmp.i = icmp ugt i32 %37, 4999
  br i1 %cmp.i, label %if.then.i, label %if.end.i

while.body:                                       ; preds = %while.body.lr.ph, %while.cond.backedge
  %38 = phi i8 [ %1, %while.body.lr.ph ], [ %43, %while.cond.backedge ]
  %pchWord.pn = phi i8* [ %pchWord, %while.body.lr.ph ], [ %incdec.ptr92, %while.cond.backedge ]
  %cchLength.091 = phi i32 [ 0, %while.body.lr.ph ], [ %cchLength.0.be, %while.cond.backedge ]
  %incdec.ptr92 = getelementptr inbounds i8, i8* %pchWord.pn, i64 1
  %idxprom = sext i8 %38 to i64
  %arrayidx = getelementptr inbounds i16, i16* %2, i64 %idxprom
  %39 = load i16, i16* %arrayidx, align 2, !tbaa !11
  %40 = and i16 %39, 1024
  %tobool.not = icmp eq i16 %40, 0
  br i1 %tobool.not, label %while.cond.backedge, label %if.end12

if.end12:                                         ; preds = %while.body
  %call.i = tail call i32** @__ctype_tolower_loc() #16
  %.pn = load i32*, i32** %call.i, align 8, !tbaa !2
  %__res.0.in = getelementptr inbounds i32, i32* %.pn, i64 %idxprom
  %__res.0 = load i32, i32* %__res.0.in, align 4, !tbaa !14
  %sub = add nsw i32 %__res.0, -97
  %idxprom13 = sext i32 %sub to i64
  %arrayidx14 = getelementptr inbounds [26 x i8], [26 x i8]* %cchFrequency, i64 0, i64 %idxprom13
  %41 = load i8, i8* %arrayidx14, align 1, !tbaa !13
  %inc = add i8 %41, 1
  store i8 %inc, i8* %arrayidx14, align 1, !tbaa !13
  %conv15 = zext i8 %inc to i32
  %uFrequency = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %idxprom13, i32 0
  %42 = load i32, i32* %uFrequency, align 16, !tbaa !15
  %cmp18 = icmp ult i32 %42, %conv15
  br i1 %cmp18, label %cleanup, label %if.end21

if.end21:                                         ; preds = %if.end12
  %inc22 = add nsw i32 %cchLength.091, 1
  br label %while.cond.backedge

while.cond.backedge:                              ; preds = %if.end21, %while.body
  %cchLength.0.be = phi i32 [ %inc22, %if.end21 ], [ %cchLength.091, %while.body ]
  %43 = load i8, i8* %incdec.ptr92, align 1, !tbaa !13
  %cmp.not = icmp eq i8 %43, 0
  br i1 %cmp.not, label %for.cond.preheader.loopexit, label %while.body

if.then.i:                                        ; preds = %for.cond.preheader
  tail call void @Fatal(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.10, i64 0, i64 0), i32 0) #15
  %.pre.i = load i32, i32* @cpwCand, align 4, !tbaa !14
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.cond.preheader
  %44 = phi i32 [ %.pre.i, %if.then.i ], [ %37, %for.cond.preheader ]
  %inc.i = add i32 %44, 1
  store i32 %inc.i, i32* @cpwCand, align 4, !tbaa !14
  %idxprom.i84 = zext i32 %44 to i64
  %arrayidx.i85 = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %idxprom.i84
  %45 = load %struct.Word*, %struct.Word** %arrayidx.i85, align 8, !tbaa !2
  %cmp1.not.i = icmp eq %struct.Word* %45, null
  br i1 %cmp1.not.i, label %if.end3.i, label %NextWord.exit

if.end3.i:                                        ; preds = %if.end.i
  %call.i.i = tail call noalias dereferenceable_or_null(32) i8* @malloc(i64 32) #15
  %cmp.i.i = icmp eq i8* %call.i.i, null
  br i1 %cmp.i.i, label %if.then.i.i, label %NewWord.exit.i

if.then.i.i:                                      ; preds = %if.end3.i
  tail call void @Fatal(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.8, i64 0, i64 0), i32 %inc.i) #15
  %.pre11.i = load i32, i32* @cpwCand, align 4, !tbaa !14
  br label %NewWord.exit.i

NewWord.exit.i:                                   ; preds = %if.then.i.i, %if.end3.i
  %46 = phi i32 [ %inc.i, %if.end3.i ], [ %.pre11.i, %if.then.i.i ]
  %47 = bitcast i8* %call.i.i to %struct.Word*
  %sub.i = add i32 %46, -1
  %idxprom4.i = zext i32 %sub.i to i64
  %arrayidx5.i = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %idxprom4.i
  %48 = bitcast %struct.Word** %arrayidx5.i to i8**
  store i8* %call.i.i, i8** %48, align 8, !tbaa !2
  br label %NextWord.exit

NextWord.exit:                                    ; preds = %if.end.i, %NewWord.exit.i
  %retval.0.i = phi %struct.Word* [ %47, %NewWord.exit.i ], [ %45, %if.end.i ]
  %49 = bitcast %struct.Word* %retval.0.i to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %49, i8 0, i64 16, i1 false)
  %pchWord33 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 1
  store i8* %pchWord, i8** %pchWord33, align 8, !tbaa !21
  %cchLength34 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 2
  store i32 %cchLength.0.lcssa, i32* %cchLength34, align 8, !tbaa !23
  %50 = load i8, i8* %0, align 16, !tbaa !13
  %conv41 = zext i8 %50 to i64
  %51 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 0, i32 1), align 4, !tbaa !19
  %sh_prom = zext i32 %51 to i64
  %shl = shl i64 %conv41, %sh_prom
  %52 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 0, i32 3), align 4, !tbaa !20
  %idxprom47 = zext i32 %52 to i64
  %arrayidx48 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47
  %53 = load i64, i64* %arrayidx48, align 8, !tbaa !18
  %or = or i64 %53, %shl
  store i64 %or, i64* %arrayidx48, align 8, !tbaa !18
  %54 = load i8, i8* %arrayidx26.1, align 1, !tbaa !13
  %conv41.1 = zext i8 %54 to i64
  %55 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 1, i32 1), align 4, !tbaa !19
  %sh_prom.1 = zext i32 %55 to i64
  %shl.1 = shl i64 %conv41.1, %sh_prom.1
  %56 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 1, i32 3), align 4, !tbaa !20
  %idxprom47.1 = zext i32 %56 to i64
  %arrayidx48.1 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.1
  %57 = load i64, i64* %arrayidx48.1, align 8, !tbaa !18
  %or.1 = or i64 %57, %shl.1
  store i64 %or.1, i64* %arrayidx48.1, align 8, !tbaa !18
  %58 = load i8, i8* %arrayidx26.2, align 2, !tbaa !13
  %conv41.2 = zext i8 %58 to i64
  %59 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 2, i32 1), align 4, !tbaa !19
  %sh_prom.2 = zext i32 %59 to i64
  %shl.2 = shl i64 %conv41.2, %sh_prom.2
  %60 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 2, i32 3), align 4, !tbaa !20
  %idxprom47.2 = zext i32 %60 to i64
  %arrayidx48.2 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.2
  %61 = load i64, i64* %arrayidx48.2, align 8, !tbaa !18
  %or.2 = or i64 %61, %shl.2
  store i64 %or.2, i64* %arrayidx48.2, align 8, !tbaa !18
  %62 = load i8, i8* %arrayidx26.3, align 1, !tbaa !13
  %conv41.3 = zext i8 %62 to i64
  %63 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 3, i32 1), align 4, !tbaa !19
  %sh_prom.3 = zext i32 %63 to i64
  %shl.3 = shl i64 %conv41.3, %sh_prom.3
  %64 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 3, i32 3), align 4, !tbaa !20
  %idxprom47.3 = zext i32 %64 to i64
  %arrayidx48.3 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.3
  %65 = load i64, i64* %arrayidx48.3, align 8, !tbaa !18
  %or.3 = or i64 %65, %shl.3
  store i64 %or.3, i64* %arrayidx48.3, align 8, !tbaa !18
  %66 = load i8, i8* %arrayidx26.4, align 4, !tbaa !13
  %conv41.4 = zext i8 %66 to i64
  %67 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 4, i32 1), align 4, !tbaa !19
  %sh_prom.4 = zext i32 %67 to i64
  %shl.4 = shl i64 %conv41.4, %sh_prom.4
  %68 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 4, i32 3), align 4, !tbaa !20
  %idxprom47.4 = zext i32 %68 to i64
  %arrayidx48.4 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.4
  %69 = load i64, i64* %arrayidx48.4, align 8, !tbaa !18
  %or.4 = or i64 %69, %shl.4
  store i64 %or.4, i64* %arrayidx48.4, align 8, !tbaa !18
  %70 = load i8, i8* %arrayidx26.5, align 1, !tbaa !13
  %conv41.5 = zext i8 %70 to i64
  %71 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 5, i32 1), align 4, !tbaa !19
  %sh_prom.5 = zext i32 %71 to i64
  %shl.5 = shl i64 %conv41.5, %sh_prom.5
  %72 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 5, i32 3), align 4, !tbaa !20
  %idxprom47.5 = zext i32 %72 to i64
  %arrayidx48.5 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.5
  %73 = load i64, i64* %arrayidx48.5, align 8, !tbaa !18
  %or.5 = or i64 %73, %shl.5
  store i64 %or.5, i64* %arrayidx48.5, align 8, !tbaa !18
  %74 = load i8, i8* %arrayidx26.6, align 2, !tbaa !13
  %conv41.6 = zext i8 %74 to i64
  %75 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 6, i32 1), align 4, !tbaa !19
  %sh_prom.6 = zext i32 %75 to i64
  %shl.6 = shl i64 %conv41.6, %sh_prom.6
  %76 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 6, i32 3), align 4, !tbaa !20
  %idxprom47.6 = zext i32 %76 to i64
  %arrayidx48.6 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.6
  %77 = load i64, i64* %arrayidx48.6, align 8, !tbaa !18
  %or.6 = or i64 %77, %shl.6
  store i64 %or.6, i64* %arrayidx48.6, align 8, !tbaa !18
  %78 = load i8, i8* %arrayidx26.7, align 1, !tbaa !13
  %conv41.7 = zext i8 %78 to i64
  %79 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 7, i32 1), align 4, !tbaa !19
  %sh_prom.7 = zext i32 %79 to i64
  %shl.7 = shl i64 %conv41.7, %sh_prom.7
  %80 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 7, i32 3), align 4, !tbaa !20
  %idxprom47.7 = zext i32 %80 to i64
  %arrayidx48.7 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.7
  %81 = load i64, i64* %arrayidx48.7, align 8, !tbaa !18
  %or.7 = or i64 %81, %shl.7
  store i64 %or.7, i64* %arrayidx48.7, align 8, !tbaa !18
  %82 = load i8, i8* %arrayidx26.8, align 8, !tbaa !13
  %conv41.8 = zext i8 %82 to i64
  %83 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 8, i32 1), align 4, !tbaa !19
  %sh_prom.8 = zext i32 %83 to i64
  %shl.8 = shl i64 %conv41.8, %sh_prom.8
  %84 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 8, i32 3), align 4, !tbaa !20
  %idxprom47.8 = zext i32 %84 to i64
  %arrayidx48.8 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.8
  %85 = load i64, i64* %arrayidx48.8, align 8, !tbaa !18
  %or.8 = or i64 %85, %shl.8
  store i64 %or.8, i64* %arrayidx48.8, align 8, !tbaa !18
  %86 = load i8, i8* %arrayidx26.9, align 1, !tbaa !13
  %conv41.9 = zext i8 %86 to i64
  %87 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 9, i32 1), align 4, !tbaa !19
  %sh_prom.9 = zext i32 %87 to i64
  %shl.9 = shl i64 %conv41.9, %sh_prom.9
  %88 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 9, i32 3), align 4, !tbaa !20
  %idxprom47.9 = zext i32 %88 to i64
  %arrayidx48.9 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.9
  %89 = load i64, i64* %arrayidx48.9, align 8, !tbaa !18
  %or.9 = or i64 %89, %shl.9
  store i64 %or.9, i64* %arrayidx48.9, align 8, !tbaa !18
  %90 = load i8, i8* %arrayidx26.10, align 2, !tbaa !13
  %conv41.10 = zext i8 %90 to i64
  %91 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 10, i32 1), align 4, !tbaa !19
  %sh_prom.10 = zext i32 %91 to i64
  %shl.10 = shl i64 %conv41.10, %sh_prom.10
  %92 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 10, i32 3), align 4, !tbaa !20
  %idxprom47.10 = zext i32 %92 to i64
  %arrayidx48.10 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.10
  %93 = load i64, i64* %arrayidx48.10, align 8, !tbaa !18
  %or.10 = or i64 %93, %shl.10
  store i64 %or.10, i64* %arrayidx48.10, align 8, !tbaa !18
  %94 = load i8, i8* %arrayidx26.11, align 1, !tbaa !13
  %conv41.11 = zext i8 %94 to i64
  %95 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 11, i32 1), align 4, !tbaa !19
  %sh_prom.11 = zext i32 %95 to i64
  %shl.11 = shl i64 %conv41.11, %sh_prom.11
  %96 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 11, i32 3), align 4, !tbaa !20
  %idxprom47.11 = zext i32 %96 to i64
  %arrayidx48.11 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.11
  %97 = load i64, i64* %arrayidx48.11, align 8, !tbaa !18
  %or.11 = or i64 %97, %shl.11
  store i64 %or.11, i64* %arrayidx48.11, align 8, !tbaa !18
  %98 = load i8, i8* %arrayidx26.12, align 4, !tbaa !13
  %conv41.12 = zext i8 %98 to i64
  %99 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 12, i32 1), align 4, !tbaa !19
  %sh_prom.12 = zext i32 %99 to i64
  %shl.12 = shl i64 %conv41.12, %sh_prom.12
  %100 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 12, i32 3), align 4, !tbaa !20
  %idxprom47.12 = zext i32 %100 to i64
  %arrayidx48.12 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.12
  %101 = load i64, i64* %arrayidx48.12, align 8, !tbaa !18
  %or.12 = or i64 %101, %shl.12
  store i64 %or.12, i64* %arrayidx48.12, align 8, !tbaa !18
  %102 = load i8, i8* %arrayidx26.13, align 1, !tbaa !13
  %conv41.13 = zext i8 %102 to i64
  %103 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 13, i32 1), align 4, !tbaa !19
  %sh_prom.13 = zext i32 %103 to i64
  %shl.13 = shl i64 %conv41.13, %sh_prom.13
  %104 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 13, i32 3), align 4, !tbaa !20
  %idxprom47.13 = zext i32 %104 to i64
  %arrayidx48.13 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.13
  %105 = load i64, i64* %arrayidx48.13, align 8, !tbaa !18
  %or.13 = or i64 %105, %shl.13
  store i64 %or.13, i64* %arrayidx48.13, align 8, !tbaa !18
  %106 = load i8, i8* %arrayidx26.14, align 2, !tbaa !13
  %conv41.14 = zext i8 %106 to i64
  %107 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 14, i32 1), align 4, !tbaa !19
  %sh_prom.14 = zext i32 %107 to i64
  %shl.14 = shl i64 %conv41.14, %sh_prom.14
  %108 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 14, i32 3), align 4, !tbaa !20
  %idxprom47.14 = zext i32 %108 to i64
  %arrayidx48.14 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.14
  %109 = load i64, i64* %arrayidx48.14, align 8, !tbaa !18
  %or.14 = or i64 %109, %shl.14
  store i64 %or.14, i64* %arrayidx48.14, align 8, !tbaa !18
  %110 = load i8, i8* %arrayidx26.15, align 1, !tbaa !13
  %conv41.15 = zext i8 %110 to i64
  %111 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 15, i32 1), align 4, !tbaa !19
  %sh_prom.15 = zext i32 %111 to i64
  %shl.15 = shl i64 %conv41.15, %sh_prom.15
  %112 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 15, i32 3), align 4, !tbaa !20
  %idxprom47.15 = zext i32 %112 to i64
  %arrayidx48.15 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.15
  %113 = load i64, i64* %arrayidx48.15, align 8, !tbaa !18
  %or.15 = or i64 %113, %shl.15
  store i64 %or.15, i64* %arrayidx48.15, align 8, !tbaa !18
  %114 = load i8, i8* %arrayidx26.16, align 16, !tbaa !13
  %conv41.16 = zext i8 %114 to i64
  %115 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 16, i32 1), align 4, !tbaa !19
  %sh_prom.16 = zext i32 %115 to i64
  %shl.16 = shl i64 %conv41.16, %sh_prom.16
  %116 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 16, i32 3), align 4, !tbaa !20
  %idxprom47.16 = zext i32 %116 to i64
  %arrayidx48.16 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.16
  %117 = load i64, i64* %arrayidx48.16, align 8, !tbaa !18
  %or.16 = or i64 %117, %shl.16
  store i64 %or.16, i64* %arrayidx48.16, align 8, !tbaa !18
  %118 = load i8, i8* %arrayidx26.17, align 1, !tbaa !13
  %conv41.17 = zext i8 %118 to i64
  %119 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 17, i32 1), align 4, !tbaa !19
  %sh_prom.17 = zext i32 %119 to i64
  %shl.17 = shl i64 %conv41.17, %sh_prom.17
  %120 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 17, i32 3), align 4, !tbaa !20
  %idxprom47.17 = zext i32 %120 to i64
  %arrayidx48.17 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.17
  %121 = load i64, i64* %arrayidx48.17, align 8, !tbaa !18
  %or.17 = or i64 %121, %shl.17
  store i64 %or.17, i64* %arrayidx48.17, align 8, !tbaa !18
  %122 = load i8, i8* %arrayidx26.18, align 2, !tbaa !13
  %conv41.18 = zext i8 %122 to i64
  %123 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 18, i32 1), align 4, !tbaa !19
  %sh_prom.18 = zext i32 %123 to i64
  %shl.18 = shl i64 %conv41.18, %sh_prom.18
  %124 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 18, i32 3), align 4, !tbaa !20
  %idxprom47.18 = zext i32 %124 to i64
  %arrayidx48.18 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.18
  %125 = load i64, i64* %arrayidx48.18, align 8, !tbaa !18
  %or.18 = or i64 %125, %shl.18
  store i64 %or.18, i64* %arrayidx48.18, align 8, !tbaa !18
  %126 = load i8, i8* %arrayidx26.19, align 1, !tbaa !13
  %conv41.19 = zext i8 %126 to i64
  %127 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 19, i32 1), align 4, !tbaa !19
  %sh_prom.19 = zext i32 %127 to i64
  %shl.19 = shl i64 %conv41.19, %sh_prom.19
  %128 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 19, i32 3), align 4, !tbaa !20
  %idxprom47.19 = zext i32 %128 to i64
  %arrayidx48.19 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.19
  %129 = load i64, i64* %arrayidx48.19, align 8, !tbaa !18
  %or.19 = or i64 %129, %shl.19
  store i64 %or.19, i64* %arrayidx48.19, align 8, !tbaa !18
  %130 = load i8, i8* %arrayidx26.20, align 4, !tbaa !13
  %conv41.20 = zext i8 %130 to i64
  %131 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 20, i32 1), align 4, !tbaa !19
  %sh_prom.20 = zext i32 %131 to i64
  %shl.20 = shl i64 %conv41.20, %sh_prom.20
  %132 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 20, i32 3), align 4, !tbaa !20
  %idxprom47.20 = zext i32 %132 to i64
  %arrayidx48.20 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.20
  %133 = load i64, i64* %arrayidx48.20, align 8, !tbaa !18
  %or.20 = or i64 %133, %shl.20
  store i64 %or.20, i64* %arrayidx48.20, align 8, !tbaa !18
  %134 = load i8, i8* %arrayidx26.21, align 1, !tbaa !13
  %conv41.21 = zext i8 %134 to i64
  %135 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 21, i32 1), align 4, !tbaa !19
  %sh_prom.21 = zext i32 %135 to i64
  %shl.21 = shl i64 %conv41.21, %sh_prom.21
  %136 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 21, i32 3), align 4, !tbaa !20
  %idxprom47.21 = zext i32 %136 to i64
  %arrayidx48.21 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.21
  %137 = load i64, i64* %arrayidx48.21, align 8, !tbaa !18
  %or.21 = or i64 %137, %shl.21
  store i64 %or.21, i64* %arrayidx48.21, align 8, !tbaa !18
  %138 = load i8, i8* %arrayidx26.22, align 2, !tbaa !13
  %conv41.22 = zext i8 %138 to i64
  %139 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 22, i32 1), align 4, !tbaa !19
  %sh_prom.22 = zext i32 %139 to i64
  %shl.22 = shl i64 %conv41.22, %sh_prom.22
  %140 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 22, i32 3), align 4, !tbaa !20
  %idxprom47.22 = zext i32 %140 to i64
  %arrayidx48.22 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.22
  %141 = load i64, i64* %arrayidx48.22, align 8, !tbaa !18
  %or.22 = or i64 %141, %shl.22
  store i64 %or.22, i64* %arrayidx48.22, align 8, !tbaa !18
  %142 = load i8, i8* %arrayidx26.23, align 1, !tbaa !13
  %conv41.23 = zext i8 %142 to i64
  %143 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 23, i32 1), align 4, !tbaa !19
  %sh_prom.23 = zext i32 %143 to i64
  %shl.23 = shl i64 %conv41.23, %sh_prom.23
  %144 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 23, i32 3), align 4, !tbaa !20
  %idxprom47.23 = zext i32 %144 to i64
  %arrayidx48.23 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.23
  %145 = load i64, i64* %arrayidx48.23, align 8, !tbaa !18
  %or.23 = or i64 %145, %shl.23
  store i64 %or.23, i64* %arrayidx48.23, align 8, !tbaa !18
  %146 = load i8, i8* %arrayidx26.24, align 8, !tbaa !13
  %conv41.24 = zext i8 %146 to i64
  %147 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 24, i32 1), align 4, !tbaa !19
  %sh_prom.24 = zext i32 %147 to i64
  %shl.24 = shl i64 %conv41.24, %sh_prom.24
  %148 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 24, i32 3), align 4, !tbaa !20
  %idxprom47.24 = zext i32 %148 to i64
  %arrayidx48.24 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.24
  %149 = load i64, i64* %arrayidx48.24, align 8, !tbaa !18
  %or.24 = or i64 %149, %shl.24
  store i64 %or.24, i64* %arrayidx48.24, align 8, !tbaa !18
  %150 = load i8, i8* %arrayidx26.25, align 1, !tbaa !13
  %conv41.25 = zext i8 %150 to i64
  %151 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 25, i32 1), align 4, !tbaa !19
  %sh_prom.25 = zext i32 %151 to i64
  %shl.25 = shl i64 %conv41.25, %sh_prom.25
  %152 = load i32, i32* getelementptr inbounds ([26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 25, i32 3), align 4, !tbaa !20
  %idxprom47.25 = zext i32 %152 to i64
  %arrayidx48.25 = getelementptr inbounds %struct.Word, %struct.Word* %retval.0.i, i64 0, i32 0, i64 %idxprom47.25
  %153 = load i64, i64* %arrayidx48.25, align 8, !tbaa !18
  %or.25 = or i64 %153, %shl.25
  store i64 %or.25, i64* %arrayidx48.25, align 8, !tbaa !18
  br label %cleanup

cleanup:                                          ; preds = %if.end12, %NextWord.exit
  call void @llvm.lifetime.end.p0i8(i64 26, i8* nonnull %0) #15
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @AddWords() local_unnamed_addr #3 {
entry:
  %0 = load i8*, i8** @pchDictionary, align 8, !tbaa !2
  store i32 0, i32* @cpwCand, align 4, !tbaa !14
  %1 = load i8, i8* %0, align 1, !tbaa !13
  %tobool.not19 = icmp eq i8 %1, 0
  br i1 %tobool.not19, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %if.end
  %2 = phi i8 [ %8, %if.end ], [ %1, %entry ]
  %pch.020 = phi i8* [ %add.ptr11, %if.end ], [ %0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %pch.020, i64 1
  %3 = load i8, i8* %arrayidx, align 1, !tbaa !13
  %conv = sext i8 %3 to i32
  %4 = load i32, i32* @cchMinLength, align 4, !tbaa !14
  %cmp.not = icmp sgt i32 %4, %conv
  br i1 %cmp.not, label %while.body.lor.lhs.false_crit_edge, label %land.lhs.true

while.body.lor.lhs.false_crit_edge:               ; preds = %while.body
  %.pre = load i32, i32* @cchPhraseLength, align 4, !tbaa !14
  br label %lor.lhs.false

land.lhs.true:                                    ; preds = %while.body
  %add = add nsw i32 %4, %conv
  %5 = load i32, i32* @cchPhraseLength, align 4, !tbaa !14
  %cmp4.not = icmp sgt i32 %add, %5
  br i1 %cmp4.not, label %lor.lhs.false, label %if.then

lor.lhs.false:                                    ; preds = %while.body.lor.lhs.false_crit_edge, %land.lhs.true
  %6 = phi i32 [ %.pre, %while.body.lor.lhs.false_crit_edge ], [ %5, %land.lhs.true ]
  %cmp8 = icmp eq i32 %6, %conv
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %land.lhs.true
  %add.ptr = getelementptr inbounds i8, i8* %pch.020, i64 2
  tail call void @BuildWord(i8* nonnull %add.ptr)
  %.pre21 = load i8, i8* %pch.020, align 1, !tbaa !13
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false
  %7 = phi i8 [ %.pre21, %if.then ], [ %2, %lor.lhs.false ]
  %idx.ext = sext i8 %7 to i64
  %add.ptr11 = getelementptr inbounds i8, i8* %pch.020, i64 %idx.ext
  %8 = load i8, i8* %add.ptr11, align 1, !tbaa !13
  %tobool.not = icmp eq i8 %8, 0
  br i1 %tobool.not, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %if.end
  %.pre22 = load i32, i32* @cpwCand, align 4, !tbaa !14
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %9 = phi i32 [ %.pre22, %while.end.loopexit ], [ 0, %entry ]
  %10 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %10, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.11, i64 0, i64 0), i32 %9) #13
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @DumpCandidates() local_unnamed_addr #7 {
entry:
  %0 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %cmp7.not = icmp eq i32 %0, 0
  br i1 %cmp7.not, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %indvars.iv
  %1 = load %struct.Word*, %struct.Word** %arrayidx, align 8, !tbaa !2
  %pchWord = getelementptr inbounds %struct.Word, %struct.Word* %1, i64 0, i32 1
  %2 = load i8*, i8** %pchWord, align 8, !tbaa !21
  %rem9 = and i64 %indvars.iv, 3
  %cmp1 = icmp eq i64 %rem9, 3
  %cond = select i1 %cmp1, i32 10, i32 32
  %call = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([7 x i8], [7 x i8]* @.str.12, i64 0, i64 0), i8* %2, i32 %cond)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %3 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %4 = zext i32 %3 to i64
  %cmp = icmp ult i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  %putchar = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @DumpWords() local_unnamed_addr #7 {
entry:
  %0 = load i32, i32* @DumpWords.X, align 4, !tbaa !14
  %add = add nsw i32 %0, 1
  %and = and i32 %add, 1023
  store i32 %and, i32* @DumpWords.X, align 4, !tbaa !14
  %cmp.not = icmp eq i32 %and, 0
  br i1 %cmp.not, label %for.cond.preheader, label %cleanup

for.cond.preheader:                               ; preds = %entry
  %1 = load i32, i32* @cpwLast, align 4, !tbaa !14
  %cmp15 = icmp sgt i32 %1, 0
  br i1 %cmp15, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.cond.preheader ]
  %arrayidx = getelementptr inbounds [51 x %struct.Word*], [51 x %struct.Word*]* @apwSol, i64 0, i64 %indvars.iv
  %2 = load %struct.Word*, %struct.Word** %arrayidx, align 8, !tbaa !2
  %pchWord = getelementptr inbounds %struct.Word, %struct.Word* %2, i64 0, i32 1
  %3 = load i8*, i8** %pchWord, align 8, !tbaa !21
  %call.i = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), i8* %3) #15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %4 = load i32, i32* @cpwLast, align 4, !tbaa !14
  %5 = sext i32 %4 to i64
  %cmp1 = icmp slt i64 %indvars.iv.next, %5
  br i1 %cmp1, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %for.cond.preheader
  %putchar = tail call i32 @putchar(i32 10)
  br label %cleanup

cleanup:                                          ; preds = %entry, %for.end
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @FindAnagram(i64* nocapture readonly %pqMask, %struct.Word** %ppwStart, i32 %iLetter) local_unnamed_addr #3 {
entry:
  %aqNext = alloca [2 x i64], align 16
  %0 = bitcast [2 x i64]* %aqNext to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #15
  %1 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %idx.ext = zext i32 %1 to i64
  %2 = sext i32 %iLetter to i64
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.cond ], [ %2, %entry ]
  %arrayidx = getelementptr inbounds [26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 %indvars.iv
  %3 = load i8, i8* %arrayidx, align 1, !tbaa !13
  %idxprom1 = sext i8 %3 to i64
  %iq3 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %idxprom1, i32 3
  %4 = load i32, i32* %iq3, align 4, !tbaa !20
  %uBits = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %idxprom1, i32 2
  %5 = load i32, i32* %uBits, align 8, !tbaa !17
  %uShift = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %idxprom1, i32 1
  %6 = load i32, i32* %uShift, align 4, !tbaa !19
  %shl = shl i32 %5, %6
  %conv = zext i32 %shl to i64
  %idxprom12 = zext i32 %4 to i64
  %arrayidx13 = getelementptr inbounds i64, i64* %pqMask, i64 %idxprom12
  %7 = load i64, i64* %arrayidx13, align 8, !tbaa !18
  %and = and i64 %7, %conv
  %tobool.not = icmp eq i64 %and, 0
  %indvars.iv.next = add i64 %indvars.iv, 1
  br i1 %tobool.not, label %for.cond, label %while.cond.preheader

while.cond.preheader:                             ; preds = %for.cond
  %conv.le = zext i32 %shl to i64
  %idxprom12.le = zext i32 %4 to i64
  %add.ptr = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %idx.ext
  %8 = trunc i64 %indvars.iv to i32
  %cmp8896 = icmp ugt %struct.Word** %add.ptr, %ppwStart
  br i1 %cmp8896, label %while.body.lr.ph.lr.ph, label %while.end

while.body.lr.ph.lr.ph:                           ; preds = %while.cond.preheader
  %arrayidx17 = getelementptr inbounds [2 x i64], [2 x i64]* %aqNext, i64 0, i64 0
  %arrayidx22 = getelementptr inbounds i64, i64* %pqMask, i64 1
  %arrayidx26 = getelementptr inbounds [2 x i64], [2 x i64]* %aqNext, i64 0, i64 1
  br label %while.body.lr.ph

while.body:                                       ; preds = %if.then38
  %9 = load %struct.Word*, %struct.Word** %ppwStart.addr.0.ph97, align 8, !tbaa !2
  %arrayidx16 = getelementptr inbounds %struct.Word, %struct.Word* %9, i64 0, i32 0, i64 0
  %10 = load i64, i64* %arrayidx16, align 8, !tbaa !18
  %sub = sub i64 %11, %10
  %and18 = and i64 %12, %sub
  %tobool19.not = icmp eq i64 %and18, 0
  br i1 %tobool19.not, label %if.end21, label %if.then20

if.then20:                                        ; preds = %while.body, %while.body.lr.ph
  %sub2593.lcssa = phi i64 [ %arrayidx26.promoted, %while.body.lr.ph ], [ %sub25, %while.body ]
  %ppwEnd.089.lcssa = phi %struct.Word** [ %ppwEnd.0.ph98, %while.body.lr.ph ], [ %incdec.ptr39, %while.body ]
  %sub.lcssa = phi i64 [ %sub144, %while.body.lr.ph ], [ %sub, %while.body ]
  store i64 %sub.lcssa, i64* %arrayidx17, align 16, !tbaa !18
  store i64 %sub2593.lcssa, i64* %arrayidx26, align 8, !tbaa !18
  br label %while.cond.outer.backedge

while.cond.outer.backedge:                        ; preds = %if.then20, %if.then29, %if.end50
  %arrayidx26.promoted114 = phi i64 [ %sub25, %if.end50 ], [ %sub25, %if.then29 ], [ %sub2593.lcssa, %if.then20 ]
  %ppwEnd.0.ph.be = phi %struct.Word** [ %ppwEnd.1, %if.end50 ], [ %ppwEnd.089147, %if.then29 ], [ %ppwEnd.089.lcssa, %if.then20 ]
  %ppwStart.addr.0.ph.be = getelementptr inbounds %struct.Word*, %struct.Word** %ppwStart.addr.0.ph97, i64 1
  %cmp88 = icmp ult %struct.Word** %ppwStart.addr.0.ph.be, %ppwEnd.0.ph.be
  br i1 %cmp88, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %while.body.lr.ph.lr.ph, %while.cond.outer.backedge
  %arrayidx26.promoted = phi i64 [ undef, %while.body.lr.ph.lr.ph ], [ %arrayidx26.promoted114, %while.cond.outer.backedge ]
  %ppwEnd.0.ph98 = phi %struct.Word** [ %add.ptr, %while.body.lr.ph.lr.ph ], [ %ppwEnd.0.ph.be, %while.cond.outer.backedge ]
  %ppwStart.addr.0.ph97 = phi %struct.Word** [ %ppwStart, %while.body.lr.ph.lr.ph ], [ %ppwStart.addr.0.ph.be, %while.cond.outer.backedge ]
  %11 = load i64, i64* %pqMask, align 8, !tbaa !18
  %12 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @aqMainSign, i64 0, i64 0), align 16, !tbaa !18
  %13 = load i64, i64* getelementptr inbounds ([2 x i64], [2 x i64]* @aqMainSign, i64 0, i64 1), align 8
  %14 = bitcast %struct.Word** %ppwStart.addr.0.ph97 to i64*
  %15 = load %struct.Word*, %struct.Word** %ppwStart.addr.0.ph97, align 8, !tbaa !2
  %arrayidx16143 = getelementptr inbounds %struct.Word, %struct.Word* %15, i64 0, i32 0, i64 0
  %16 = load i64, i64* %arrayidx16143, align 8, !tbaa !18
  %sub144 = sub i64 %11, %16
  %and18145 = and i64 %12, %sub144
  %tobool19.not146 = icmp eq i64 %and18145, 0
  br i1 %tobool19.not146, label %if.end21.preheader, label %if.then20

if.end21.preheader:                               ; preds = %while.body.lr.ph
  %17 = load i64, i64* %arrayidx22, align 8, !tbaa !18
  br label %if.end21

if.end21:                                         ; preds = %if.end21.preheader, %while.body
  %sub148 = phi i64 [ %sub, %while.body ], [ %sub144, %if.end21.preheader ]
  %18 = phi %struct.Word* [ %9, %while.body ], [ %15, %if.end21.preheader ]
  %ppwEnd.089147 = phi %struct.Word** [ %incdec.ptr39, %while.body ], [ %ppwEnd.0.ph98, %if.end21.preheader ]
  %arrayidx24 = getelementptr inbounds %struct.Word, %struct.Word* %18, i64 0, i32 0, i64 1
  %19 = load i64, i64* %arrayidx24, align 8, !tbaa !18
  %sub25 = sub i64 %17, %19
  %and27 = and i64 %13, %sub25
  %tobool28.not = icmp eq i64 %and27, 0
  br i1 %tobool28.not, label %if.end31, label %if.then29

if.then29:                                        ; preds = %if.end21
  store i64 %sub148, i64* %arrayidx17, align 16, !tbaa !18
  store i64 %sub25, i64* %arrayidx26, align 8, !tbaa !18
  br label %while.cond.outer.backedge

if.end31:                                         ; preds = %if.end21
  %arrayidx34 = getelementptr inbounds %struct.Word, %struct.Word* %18, i64 0, i32 0, i64 %idxprom12.le
  %20 = load i64, i64* %arrayidx34, align 8, !tbaa !18
  %and35 = and i64 %20, %conv.le
  %cmp36 = icmp eq i64 %and35, 0
  br i1 %cmp36, label %if.then38, label %if.end40

if.then38:                                        ; preds = %if.end31
  %incdec.ptr39 = getelementptr inbounds %struct.Word*, %struct.Word** %ppwEnd.089147, i64 -1
  %21 = bitcast %struct.Word** %incdec.ptr39 to i64*
  %22 = load i64, i64* %21, align 8, !tbaa !2
  store i64 %22, i64* %14, align 8, !tbaa !2
  store %struct.Word* %18, %struct.Word** %incdec.ptr39, align 8, !tbaa !2
  %cmp = icmp ult %struct.Word** %ppwStart.addr.0.ph97, %incdec.ptr39
  br i1 %cmp, label %while.body, label %while.cond.while.end_crit_edge

if.end40:                                         ; preds = %if.end31
  store i64 %sub148, i64* %arrayidx17, align 16, !tbaa !18
  store i64 %sub25, i64* %arrayidx26, align 8, !tbaa !18
  %23 = load i32, i32* @cpwLast, align 4, !tbaa !14
  %inc41 = add nsw i32 %23, 1
  store i32 %inc41, i32* @cpwLast, align 4, !tbaa !14
  %idxprom42 = sext i32 %23 to i64
  %arrayidx43 = getelementptr inbounds [51 x %struct.Word*], [51 x %struct.Word*]* @apwSol, i64 0, i64 %idxprom42
  store %struct.Word* %18, %struct.Word** %arrayidx43, align 8, !tbaa !2
  %cchLength = getelementptr inbounds %struct.Word, %struct.Word* %18, i64 0, i32 2
  %24 = load i32, i32* %cchLength, align 8, !tbaa !23
  %25 = load i32, i32* @cchPhraseLength, align 4, !tbaa !14
  %sub44 = sub i32 %25, %24
  store i32 %sub44, i32* @cchPhraseLength, align 4, !tbaa !14
  %tobool45.not = icmp eq i32 %sub44, 0
  br i1 %tobool45.not, label %if.else, label %if.then46

if.then46:                                        ; preds = %if.end40
  %26 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %idx.ext47 = zext i32 %26 to i64
  %add.ptr48 = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %idx.ext47
  call void @FindAnagram(i64* nonnull %arrayidx17, %struct.Word** nonnull %ppwStart.addr.0.ph97, i32 %8)
  br label %if.end50

if.else:                                          ; preds = %if.end40
  %27 = load i32, i32* @DumpWords.X, align 4, !tbaa !14
  %add.i = add nsw i32 %27, 1
  %and.i = and i32 %add.i, 1023
  store i32 %and.i, i32* @DumpWords.X, align 4, !tbaa !14
  %cmp.not.i = icmp eq i32 %and.i, 0
  br i1 %cmp.not.i, label %for.cond.preheader.i, label %if.end50

for.cond.preheader.i:                             ; preds = %if.else
  %cmp15.i = icmp sgt i32 %23, -1
  br i1 %cmp15.i, label %for.body.i, label %for.end.i

for.body.i:                                       ; preds = %for.cond.preheader.i, %for.body.i
  %indvars.iv.i = phi i64 [ %indvars.iv.next.i, %for.body.i ], [ 0, %for.cond.preheader.i ]
  %arrayidx.i = getelementptr inbounds [51 x %struct.Word*], [51 x %struct.Word*]* @apwSol, i64 0, i64 %indvars.iv.i
  %28 = load %struct.Word*, %struct.Word** %arrayidx.i, align 8, !tbaa !2
  %pchWord.i = getelementptr inbounds %struct.Word, %struct.Word* %28, i64 0, i32 1
  %29 = load i8*, i8** %pchWord.i, align 8, !tbaa !21
  %call.i.i = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), i8* %29) #15
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %30 = load i32, i32* @cpwLast, align 4, !tbaa !14
  %31 = sext i32 %30 to i64
  %cmp1.i = icmp slt i64 %indvars.iv.next.i, %31
  br i1 %cmp1.i, label %for.body.i, label %for.end.i

for.end.i:                                        ; preds = %for.body.i, %for.cond.preheader.i
  %putchar.i = tail call i32 @putchar(i32 10) #15
  br label %if.end50

if.end50:                                         ; preds = %for.end.i, %if.else, %if.then46
  %ppwEnd.1 = phi %struct.Word** [ %add.ptr48, %if.then46 ], [ %ppwEnd.089147, %if.else ], [ %ppwEnd.089147, %for.end.i ]
  %32 = load i32, i32* %cchLength, align 8, !tbaa !23
  %33 = load i32, i32* @cchPhraseLength, align 4, !tbaa !14
  %add = add i32 %33, %32
  store i32 %add, i32* @cchPhraseLength, align 4, !tbaa !14
  %34 = load i32, i32* @cpwLast, align 4, !tbaa !14
  %dec = add nsw i32 %34, -1
  store i32 %dec, i32* @cpwLast, align 4, !tbaa !14
  br label %while.cond.outer.backedge

while.cond.while.end_crit_edge:                   ; preds = %if.then38
  store i64 %sub148, i64* %arrayidx17, align 16, !tbaa !18
  store i64 %sub25, i64* %arrayidx26, align 8, !tbaa !18
  br label %while.end

while.end:                                        ; preds = %while.cond.outer.backedge, %while.cond.preheader, %while.cond.while.end_crit_edge
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #15
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local i32 @CompareFrequency(i8* nocapture readonly %pch1, i8* nocapture readonly %pch2) #8 {
entry:
  %0 = load i8, i8* %pch1, align 1, !tbaa !13
  %idxprom = sext i8 %0 to i64
  %arrayidx = getelementptr inbounds [26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 %idxprom
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !14
  %2 = load i8, i8* %pch2, align 1, !tbaa !13
  %idxprom1 = sext i8 %2 to i64
  %arrayidx2 = getelementptr inbounds [26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 %idxprom1
  %3 = load i32, i32* %arrayidx2, align 4, !tbaa !14
  %cmp = icmp ult i32 %1, %3
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp7 = icmp ugt i32 %1, %3
  br i1 %cmp7, label %return, label %if.end9

if.end9:                                          ; preds = %if.end
  %cmp11 = icmp slt i8 %0, %2
  br i1 %cmp11, label %return, label %if.end14

if.end14:                                         ; preds = %if.end9
  %cmp17 = icmp sgt i8 %0, %2
  %spec.select = zext i1 %cmp17 to i32
  ret i32 %spec.select

return:                                           ; preds = %if.end9, %if.end, %entry
  %retval.0 = phi i32 [ -1, %entry ], [ 1, %if.end ], [ -1, %if.end9 ]
  ret i32 %retval.0
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @SortCandidates() local_unnamed_addr #7 {
entry:
  store <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, <16 x i8>* bitcast ([26 x i8]* @achByFrequency to <16 x i8>*), align 16, !tbaa !13
  store i8 16, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 16), align 16, !tbaa !13
  store i8 17, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 17), align 1, !tbaa !13
  store i8 18, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 18), align 2, !tbaa !13
  store i8 19, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 19), align 1, !tbaa !13
  store i8 20, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 20), align 4, !tbaa !13
  store i8 21, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 21), align 1, !tbaa !13
  store i8 22, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 22), align 2, !tbaa !13
  store i8 23, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 23), align 1, !tbaa !13
  store i8 24, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 24), align 8, !tbaa !13
  store i8 25, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 25), align 1, !tbaa !13
  tail call void @qsort(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 0), i64 26, i64 1, i32 (i8*, i8*)* nonnull @CompareFrequency) #15
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %1 = tail call i64 @fwrite(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.14, i64 0, i64 0), i64 24, i64 1, %struct._IO_FILE* %0) #13
  %2 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 0), align 16, !tbaa !13
  %conv7 = sext i8 %2 to i32
  %add = add nsw i32 %conv7, 97
  %3 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8 = tail call i32 @fputc(i32 %add, %struct._IO_FILE* %3)
  %4 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 1), align 1, !tbaa !13
  %conv7.1 = sext i8 %4 to i32
  %add.1 = add nsw i32 %conv7.1, 97
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.1 = tail call i32 @fputc(i32 %add.1, %struct._IO_FILE* %5)
  %6 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 2), align 2, !tbaa !13
  %conv7.2 = sext i8 %6 to i32
  %add.2 = add nsw i32 %conv7.2, 97
  %7 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.2 = tail call i32 @fputc(i32 %add.2, %struct._IO_FILE* %7)
  %8 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 3), align 1, !tbaa !13
  %conv7.3 = sext i8 %8 to i32
  %add.3 = add nsw i32 %conv7.3, 97
  %9 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.3 = tail call i32 @fputc(i32 %add.3, %struct._IO_FILE* %9)
  %10 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 4), align 4, !tbaa !13
  %conv7.4 = sext i8 %10 to i32
  %add.4 = add nsw i32 %conv7.4, 97
  %11 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.4 = tail call i32 @fputc(i32 %add.4, %struct._IO_FILE* %11)
  %12 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 5), align 1, !tbaa !13
  %conv7.5 = sext i8 %12 to i32
  %add.5 = add nsw i32 %conv7.5, 97
  %13 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.5 = tail call i32 @fputc(i32 %add.5, %struct._IO_FILE* %13)
  %14 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 6), align 2, !tbaa !13
  %conv7.6 = sext i8 %14 to i32
  %add.6 = add nsw i32 %conv7.6, 97
  %15 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.6 = tail call i32 @fputc(i32 %add.6, %struct._IO_FILE* %15)
  %16 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 7), align 1, !tbaa !13
  %conv7.7 = sext i8 %16 to i32
  %add.7 = add nsw i32 %conv7.7, 97
  %17 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.7 = tail call i32 @fputc(i32 %add.7, %struct._IO_FILE* %17)
  %18 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 8), align 8, !tbaa !13
  %conv7.8 = sext i8 %18 to i32
  %add.8 = add nsw i32 %conv7.8, 97
  %19 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.8 = tail call i32 @fputc(i32 %add.8, %struct._IO_FILE* %19)
  %20 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 9), align 1, !tbaa !13
  %conv7.9 = sext i8 %20 to i32
  %add.9 = add nsw i32 %conv7.9, 97
  %21 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.9 = tail call i32 @fputc(i32 %add.9, %struct._IO_FILE* %21)
  %22 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 10), align 2, !tbaa !13
  %conv7.10 = sext i8 %22 to i32
  %add.10 = add nsw i32 %conv7.10, 97
  %23 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.10 = tail call i32 @fputc(i32 %add.10, %struct._IO_FILE* %23)
  %24 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 11), align 1, !tbaa !13
  %conv7.11 = sext i8 %24 to i32
  %add.11 = add nsw i32 %conv7.11, 97
  %25 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.11 = tail call i32 @fputc(i32 %add.11, %struct._IO_FILE* %25)
  %26 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 12), align 4, !tbaa !13
  %conv7.12 = sext i8 %26 to i32
  %add.12 = add nsw i32 %conv7.12, 97
  %27 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.12 = tail call i32 @fputc(i32 %add.12, %struct._IO_FILE* %27)
  %28 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 13), align 1, !tbaa !13
  %conv7.13 = sext i8 %28 to i32
  %add.13 = add nsw i32 %conv7.13, 97
  %29 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.13 = tail call i32 @fputc(i32 %add.13, %struct._IO_FILE* %29)
  %30 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 14), align 2, !tbaa !13
  %conv7.14 = sext i8 %30 to i32
  %add.14 = add nsw i32 %conv7.14, 97
  %31 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.14 = tail call i32 @fputc(i32 %add.14, %struct._IO_FILE* %31)
  %32 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 15), align 1, !tbaa !13
  %conv7.15 = sext i8 %32 to i32
  %add.15 = add nsw i32 %conv7.15, 97
  %33 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.15 = tail call i32 @fputc(i32 %add.15, %struct._IO_FILE* %33)
  %34 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 16), align 16, !tbaa !13
  %conv7.16 = sext i8 %34 to i32
  %add.16 = add nsw i32 %conv7.16, 97
  %35 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.16 = tail call i32 @fputc(i32 %add.16, %struct._IO_FILE* %35)
  %36 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 17), align 1, !tbaa !13
  %conv7.17 = sext i8 %36 to i32
  %add.17 = add nsw i32 %conv7.17, 97
  %37 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.17 = tail call i32 @fputc(i32 %add.17, %struct._IO_FILE* %37)
  %38 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 18), align 2, !tbaa !13
  %conv7.18 = sext i8 %38 to i32
  %add.18 = add nsw i32 %conv7.18, 97
  %39 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.18 = tail call i32 @fputc(i32 %add.18, %struct._IO_FILE* %39)
  %40 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 19), align 1, !tbaa !13
  %conv7.19 = sext i8 %40 to i32
  %add.19 = add nsw i32 %conv7.19, 97
  %41 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.19 = tail call i32 @fputc(i32 %add.19, %struct._IO_FILE* %41)
  %42 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 20), align 4, !tbaa !13
  %conv7.20 = sext i8 %42 to i32
  %add.20 = add nsw i32 %conv7.20, 97
  %43 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.20 = tail call i32 @fputc(i32 %add.20, %struct._IO_FILE* %43)
  %44 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 21), align 1, !tbaa !13
  %conv7.21 = sext i8 %44 to i32
  %add.21 = add nsw i32 %conv7.21, 97
  %45 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.21 = tail call i32 @fputc(i32 %add.21, %struct._IO_FILE* %45)
  %46 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 22), align 2, !tbaa !13
  %conv7.22 = sext i8 %46 to i32
  %add.22 = add nsw i32 %conv7.22, 97
  %47 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.22 = tail call i32 @fputc(i32 %add.22, %struct._IO_FILE* %47)
  %48 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 23), align 1, !tbaa !13
  %conv7.23 = sext i8 %48 to i32
  %add.23 = add nsw i32 %conv7.23, 97
  %49 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.23 = tail call i32 @fputc(i32 %add.23, %struct._IO_FILE* %49)
  %50 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 24), align 8, !tbaa !13
  %conv7.24 = sext i8 %50 to i32
  %add.24 = add nsw i32 %conv7.24, 97
  %51 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.24 = tail call i32 @fputc(i32 %add.24, %struct._IO_FILE* %51)
  %52 = load i8, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @achByFrequency, i64 0, i64 25), align 1, !tbaa !13
  %conv7.25 = sext i8 %52 to i32
  %add.25 = add nsw i32 %conv7.25, 97
  %53 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call8.25 = tail call i32 @fputc(i32 %add.25, %struct._IO_FILE* %53)
  %54 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call12 = tail call i32 @fputc(i32 10, %struct._IO_FILE* %54)
  ret void
}

; Function Attrs: nofree
declare dso_local void @qsort(i8* noundef, i64 noundef, i64 noundef, i32 (i8*, i8*)* nocapture noundef) local_unnamed_addr #9

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @fputc(i32 noundef, %struct._IO_FILE* nocapture noundef) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local i8* @GetPhrase(i8* returned %pch, i32 %size) local_unnamed_addr #3 {
entry:
  %0 = load i32, i32* @fInteractive, align 4, !tbaa !14
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %putchar = tail call i32 @putchar(i32 62)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8, !tbaa !2
  %call1 = tail call i32 @fflush(%struct._IO_FILE* %1)
  %2 = load %struct._IO_FILE*, %struct._IO_FILE** @stdin, align 8, !tbaa !2
  %call2 = tail call i8* @fgets(i8* %pch, i32 %size, %struct._IO_FILE* %2)
  %cmp = icmp eq i8* %call2, null
  br i1 %cmp, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.end
  tail call void @exit(i32 0) #14
  unreachable

if.end4:                                          ; preds = %if.end
  ret i8* %pch
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @fflush(%struct._IO_FILE* nocapture noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noundef i8* @fgets(i8* noundef, i32 noundef, %struct._IO_FILE* nocapture noundef) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %cpchArgc, i8** nocapture readonly %ppchArgv) local_unnamed_addr #3 {
entry:
  %0 = and i32 %cpchArgc, -2
  %.not = icmp eq i32 %0, 2
  br i1 %.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @Fatal(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.16, i64 0, i64 0), i32 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %cmp2 = icmp eq i32 %cpchArgc, 3
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.end
  %arrayidx = getelementptr inbounds i8*, i8** %ppchArgv, i64 2
  %1 = load i8*, i8** %arrayidx, align 8, !tbaa !2
  %call.i = call i64 @strtol(i8* nocapture nonnull %1, i8** null, i32 10) #15
  %conv.i = trunc i64 %call.i to i32
  store i32 %conv.i, i32* @cchMinLength, align 4, !tbaa !14
  br label %if.end4

if.end4:                                          ; preds = %if.then3, %if.end
  %call5 = call i32 @isatty(i32 1) #15
  store i32 %call5, i32* @fInteractive, align 4, !tbaa !14
  %arrayidx6 = getelementptr inbounds i8*, i8** %ppchArgv, i64 1
  %2 = load i8*, i8** %arrayidx6, align 8, !tbaa !2
  call void @ReadDict(i8* %2)
  br label %while.cond

while.cond:                                       ; preds = %while.cond.backedge, %if.end4
  %3 = load i32, i32* @fInteractive, align 4, !tbaa !14
  %tobool.not.i = icmp eq i32 %3, 0
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %while.cond
  %putchar.i = call i32 @putchar(i32 62) #15
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %while.cond
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8, !tbaa !2
  %call1.i = call i32 @fflush(%struct._IO_FILE* %4) #15
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stdin, align 8, !tbaa !2
  %call2.i = call i8* @fgets(i8* getelementptr inbounds ([255 x i8], [255 x i8]* @achPhrase, i64 0, i64 0), i32 255, %struct._IO_FILE* %5) #15
  %cmp.i = icmp eq i8* %call2.i, null
  br i1 %cmp.i, label %if.then3.i, label %while.body

if.then3.i:                                       ; preds = %if.end.i
  call void @exit(i32 0) #14
  unreachable

while.body:                                       ; preds = %if.end.i
  %call9 = call i16** @__ctype_b_loc() #16
  %6 = load i16*, i16** %call9, align 8, !tbaa !2
  %7 = load i8, i8* getelementptr inbounds ([255 x i8], [255 x i8]* @achPhrase, i64 0, i64 0), align 16, !tbaa !13
  %idxprom = sext i8 %7 to i64
  %arrayidx10 = getelementptr inbounds i16, i16* %6, i64 %idxprom
  %8 = load i16, i16* %arrayidx10, align 2, !tbaa !11
  %9 = and i16 %8, 2048
  %tobool.not = icmp eq i16 %9, 0
  br i1 %tobool.not, label %if.else, label %if.then12

if.then12:                                        ; preds = %while.body
  %call.i37 = call i64 @strtol(i8* nocapture nonnull getelementptr inbounds ([255 x i8], [255 x i8]* @achPhrase, i64 0, i64 0), i8** null, i32 10) #15
  %conv.i38 = trunc i64 %call.i37 to i32
  store i32 %conv.i38, i32* @cchMinLength, align 4, !tbaa !14
  %call14 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([16 x i8], [16 x i8]* @.str.17, i64 0, i64 0), i32 %conv.i38)
  br label %while.cond.backedge

while.cond.backedge:                              ; preds = %if.then12, %if.end25, %if.then29, %DumpCandidates.exit, %AddWords.exit
  br label %while.cond

if.else:                                          ; preds = %while.body
  %cmp16 = icmp eq i8 %7, 63
  br i1 %cmp16, label %if.then18, label %if.else19

if.then18:                                        ; preds = %if.else
  %10 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %cmp7.not.i = icmp eq i32 %10, 0
  br i1 %cmp7.not.i, label %DumpCandidates.exit, label %for.body.i

for.body.i:                                       ; preds = %if.then18, %for.body.i
  %indvars.iv.i = phi i64 [ %indvars.iv.next.i, %for.body.i ], [ 0, %if.then18 ]
  %arrayidx.i = getelementptr inbounds [5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 %indvars.iv.i
  %11 = load %struct.Word*, %struct.Word** %arrayidx.i, align 8, !tbaa !2
  %pchWord.i = getelementptr inbounds %struct.Word, %struct.Word* %11, i64 0, i32 1
  %12 = load i8*, i8** %pchWord.i, align 8, !tbaa !21
  %rem9.i = and i64 %indvars.iv.i, 3
  %cmp1.i = icmp eq i64 %rem9.i, 3
  %cond.i = select i1 %cmp1.i, i32 10, i32 32
  %call.i39 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([7 x i8], [7 x i8]* @.str.12, i64 0, i64 0), i8* %12, i32 %cond.i) #15
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %13 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %14 = zext i32 %13 to i64
  %cmp.i40 = icmp ult i64 %indvars.iv.next.i, %14
  br i1 %cmp.i40, label %for.body.i, label %DumpCandidates.exit

DumpCandidates.exit:                              ; preds = %for.body.i, %if.then18
  %putchar.i41 = call i32 @putchar(i32 10) #15
  br label %while.cond.backedge

if.else19:                                        ; preds = %if.else
  call void @BuildMask(i8* getelementptr inbounds ([255 x i8], [255 x i8]* @achPhrase, i64 0, i64 0))
  %15 = load i8*, i8** @pchDictionary, align 8, !tbaa !2
  store i32 0, i32* @cpwCand, align 4, !tbaa !14
  %16 = load i8, i8* %15, align 1, !tbaa !13
  %tobool.not19.i = icmp eq i8 %16, 0
  br i1 %tobool.not19.i, label %AddWords.exit, label %while.body.i

while.body.i:                                     ; preds = %if.else19, %if.end.i46
  %17 = phi i8 [ %23, %if.end.i46 ], [ %16, %if.else19 ]
  %pch.020.i = phi i8* [ %add.ptr11.i, %if.end.i46 ], [ %15, %if.else19 ]
  %arrayidx.i42 = getelementptr inbounds i8, i8* %pch.020.i, i64 1
  %18 = load i8, i8* %arrayidx.i42, align 1, !tbaa !13
  %conv.i43 = sext i8 %18 to i32
  %19 = load i32, i32* @cchMinLength, align 4, !tbaa !14
  %cmp.not.i = icmp sgt i32 %19, %conv.i43
  br i1 %cmp.not.i, label %while.body.lor.lhs.false_crit_edge.i, label %land.lhs.true.i

while.body.lor.lhs.false_crit_edge.i:             ; preds = %while.body.i
  %.pre.i = load i32, i32* @cchPhraseLength, align 4, !tbaa !14
  br label %lor.lhs.false.i

land.lhs.true.i:                                  ; preds = %while.body.i
  %add.i = add nsw i32 %19, %conv.i43
  %20 = load i32, i32* @cchPhraseLength, align 4, !tbaa !14
  %cmp4.not.i = icmp sgt i32 %add.i, %20
  br i1 %cmp4.not.i, label %lor.lhs.false.i, label %if.then.i44

lor.lhs.false.i:                                  ; preds = %land.lhs.true.i, %while.body.lor.lhs.false_crit_edge.i
  %21 = phi i32 [ %.pre.i, %while.body.lor.lhs.false_crit_edge.i ], [ %20, %land.lhs.true.i ]
  %cmp8.i = icmp eq i32 %21, %conv.i43
  br i1 %cmp8.i, label %if.then.i44, label %if.end.i46

if.then.i44:                                      ; preds = %lor.lhs.false.i, %land.lhs.true.i
  %add.ptr.i = getelementptr inbounds i8, i8* %pch.020.i, i64 2
  call void @BuildWord(i8* nonnull %add.ptr.i) #15
  %.pre21.i = load i8, i8* %pch.020.i, align 1, !tbaa !13
  br label %if.end.i46

if.end.i46:                                       ; preds = %if.then.i44, %lor.lhs.false.i
  %22 = phi i8 [ %.pre21.i, %if.then.i44 ], [ %17, %lor.lhs.false.i ]
  %idx.ext.i = sext i8 %22 to i64
  %add.ptr11.i = getelementptr inbounds i8, i8* %pch.020.i, i64 %idx.ext.i
  %23 = load i8, i8* %add.ptr11.i, align 1, !tbaa !13
  %tobool.not.i45 = icmp eq i8 %23, 0
  br i1 %tobool.not.i45, label %while.end.loopexit.i, label %while.body.i

while.end.loopexit.i:                             ; preds = %if.end.i46
  %.pre22.i = load i32, i32* @cpwCand, align 4, !tbaa !14
  br label %AddWords.exit

AddWords.exit:                                    ; preds = %if.else19, %while.end.loopexit.i
  %24 = phi i32 [ %.pre22.i, %while.end.loopexit.i ], [ 0, %if.else19 ]
  %25 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call.i47 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %25, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.11, i64 0, i64 0), i32 %24) #17
  %26 = load i32, i32* @cpwCand, align 4, !tbaa !14
  %cmp20 = icmp eq i32 %26, 0
  %27 = load i32, i32* @cchPhraseLength, align 4
  %cmp22 = icmp eq i32 %27, 0
  %or.cond33 = or i1 %cmp20, %cmp22
  br i1 %or.cond33, label %while.cond.backedge, label %if.end25

if.end25:                                         ; preds = %AddWords.exit
  store i32 0, i32* @cpwLast, align 4, !tbaa !14
  call void @SortCandidates()
  %call26 = call i32 @_setjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @jbAnagram, i64 0, i64 0)) #18
  %cmp27 = icmp eq i32 %call26, 0
  br i1 %cmp27, label %if.then29, label %while.cond.backedge

if.then29:                                        ; preds = %if.end25
  call void @FindAnagram(i64* getelementptr inbounds ([2 x i64], [2 x i64]* @aqMainMask, i64 0, i64 0), %struct.Word** getelementptr inbounds ([5000 x %struct.Word*], [5000 x %struct.Word*]* @apwCand, i64 0, i64 0), i32 0)
  br label %while.cond.backedge
}

; Function Attrs: nounwind
declare dso_local i32 @isatty(i32) local_unnamed_addr #10

; Function Attrs: nounwind returns_twice
declare dso_local i32 @_setjmp(%struct.__jmp_buf_tag*) local_unnamed_addr #11

; Function Attrs: nounwind
declare dso_local i32 @__xstat(i32, i8*, %struct.stat*) local_unnamed_addr #10

; Function Attrs: nofree nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #12

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(i8* nocapture noundef, i64 noundef, i64 noundef, %struct._IO_FILE* nocapture noundef) local_unnamed_addr #12

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind willreturn }
attributes #5 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind willreturn writeonly }
attributes #7 = { nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nofree "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { nounwind returns_twice "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-cldemote,-clwb,-clzero,-enqcmd,-fma4,-gfni,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchwt1,-ptwrite,-rdpid,-serialize,-sha,-shstk,-sse4a,-tbm,-tsxldtrk,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nofree nounwind }
attributes #13 = { cold }
attributes #14 = { noreturn nounwind }
attributes #15 = { nounwind }
attributes #16 = { nounwind readnone }
attributes #17 = { cold nounwind }
attributes #18 = { nounwind returns_twice }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !8, i64 48}
!7 = !{!"stat", !8, i64 0, !8, i64 8, !8, i64 16, !9, i64 24, !9, i64 28, !9, i64 32, !9, i64 36, !8, i64 40, !8, i64 48, !8, i64 56, !8, i64 64, !10, i64 72, !10, i64 88, !10, i64 104, !4, i64 120}
!8 = !{!"long", !4, i64 0}
!9 = !{!"int", !4, i64 0}
!10 = !{!"timespec", !8, i64 0, !8, i64 8}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !4, i64 0}
!13 = !{!4, !4, i64 0}
!14 = !{!9, !9, i64 0}
!15 = !{!16, !9, i64 0}
!16 = !{!"", !9, i64 0, !9, i64 4, !9, i64 8, !9, i64 12}
!17 = !{!16, !9, i64 8}
!18 = !{!8, !8, i64 0}
!19 = !{!16, !9, i64 4}
!20 = !{!16, !9, i64 12}
!21 = !{!22, !3, i64 16}
!22 = !{!"", !4, i64 0, !3, i64 16, !9, i64 24}
!23 = !{!22, !9, i64 24}
