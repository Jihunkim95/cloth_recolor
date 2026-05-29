# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Vault Update After Experiments

**모든 실험 직후 Obsidian vault 를 업데이트한다.**

위치 (코드 repo **바깥**의 외부 Obsidian vault — git 미추적, Windows Obsidian 네이티브):
- Windows: `C:\Research\ComputerVision\4DGS_recolor\vault\`
- WSL: `/mnt/c/Research/ComputerVision/4DGS_recolor/vault/`
- 가이드라인은 그 안의 `001연구_가이드북.md`. (구 `cloth_recolor/vault/` 사본은 폐기됨 — 거기 쓰지 말 것.)

실험 (학습 / ablation / sweep / recolor 시각 검증 등) 이 끝났다고 보고하기 *전에* 다음을 수행:

1. **`vault/results/expXXX.md`** 추가 (또는 기존 exp 가 진행형이면 갱신).
   - frontmatter: `title`, `status: done|in-progress|failed`, `date`, `code`
   - 본문 섹션 (가이드북 형식 엄수): `## 질문 / ## 설정 / ## 예측 / ## 결과 / ## 예측 맞았나? / ## 다음`
   - 실험 번호는 `vault/results/` 마지막 번호 +1
   - 사용한 개념 노트는 `[[../notes/...]]` 으로 링크

2. **`vault/notes/`** 에 새 개념·기법이 도입됐다면 노트 추가.
   - frontmatter: `title`, `layer: 0|1|2|1+2`, `date`
   - 1 노트 = 1 개념. 짧게.

3. **`vault/002Daily_루틴.md`** "이번 주 기록" 에 한 줄 추가:
   `- YYYY-MM-DD (요일): <한 줄 요약> ([[results/expXXX_...]])`

4. dashboard (`003연구_대시보드.md`) 는 dataview 가 자동 갱신 — 손대지 않음.

**예외**: 단순 syntax-check, smoke-test, viewer 띄우기, 폴더 청소 같은 *비-실험* 작업은 skip.
"실험" 의 기준 = "예측을 만들고 결과 숫자/시각 으로 검증한 것".

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
