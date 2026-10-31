# Submission: Sert Team

## Links
- Repository, final commit: https://github.com/avantyur1st/soup-final-solution/commit/`<SHA>`
  <!-- После финального коммита замените <SHA> на вывод `git rev-parse HEAD`.
       COMMIT в Kaggle-ноутбуке = этот же SHA. -->
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/azeibarjanec/soup-final-submission?scriptVersionId=`<VERSION>`
  <!-- Заполните <VERSION> после Save & Run All (номер сохранённой версии). -->
- Demo video (up to 3 minutes): `<ССЫЛКА НА ВИДЕО>`
- Presentation: `<ССЫЛКА НА ПРЕЗЕНТАЦИЮ ИЛИ ФАЙЛ В ЗАЯВКЕ>`


## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | цепочку PyTorch-операций RMSNorm (5+ мелких запусков) | совокупно с bwd: нормализация |
| `rmsnorm_bwd_dx_kernel` | `solution/rmsnorm.py` | backward RMSNorm (кэш rstd вместо пересчёта) | |
| `rope_fwd_kernel` | `solution/rope.py` | PyTorch RoPE (~2,250 запусков/прогон) | совокупно с bwd |
| `rope_bwd_kernel` | `solution/rope.py` | backward RoPE | |
| `ce_tile_stats_kernel` | `solution/chunked_ce.py` | полный тензор логитов `[B·T, 151936]` + softmax/CE | потоковая CE |
| `ce_bwd_dlogits_kernel` | `solution/chunked_ce.py` | материализацию градиентов логитов; bwd только по активным токенам | |
| `swiglu_fwd_kernel` | `solution/swiglu.py` | SiLU·mul из PyTorch | слияние активаций |
| `swiglu_bwd_kernel` | `solution/swiglu.py` | backward SwiGLU | |
| `nf4_dequant_kernel` | `solution/nf4_dequant.py` | внутреннюю деквантацию bitsandbytes (не засчитывалась) + ~2.6 ГБ кэша весов | **итого 14.5%** времени шага (порог ≥ 5%) |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- QLoRA / NF4 layout (Dettmers et al., 2023) — макет NF4-деквантования; код написан с нуля и
  побитово сверен с bitsandbytes на CPU-бэкенде.
- Liger-Kernel (fused RMSNorm / SwiGLU) — идея слияния операций и rstd-кэша.
- Online softmax (Milakov & Gimelshein, 2018) и FlashAttention-стиль тайлинга — потоковая
  Cross-Entropy по блокам словаря.
Весь код в `solution/` написан с нуля; сторонний GPU-код не копировался.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 80.35 | 54.22 | **speedup x1.4819** |
| Peak GPU memory, MiB | 14,345 | 5,403 | **base/ours x2.655** |
| Eval loss (public rows) | 2.0496 | −0.152% | качество в допуске (≤ 0.5%) |

- [x] All 10 checks of `harness/check_rules.py` passed (10/10 PASS; доля собственных ядер 14.5%)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Какие операции ускорили и почему (RMSNorm, RoPE, SwiGLU, потоковая Cross-Entropy, per-use
деквантование NF4), как устроены ядра, результаты против базы (скорость x1.4819, память x2.655,
качество −0.152%), ограничения и следующие шаги.

## Limitations
- Per-use деквантование NF4 тратит время каждого шага (входит в 14.5% собственного кода);
  вариант с кэшированием деквантованных весов быстрее по wall, но требует ~2.6 ГБ VRAM и даёт
  более низкий итоговый балл по официальной формуле.
- Выигрыш `unpad.py` снижается, если в данных нет паддинга.
- Ускорение — отношение к базе той же сессии: абсолютное wall-время различается между прогонами,
  соотношение стабильно.
- Дальнейшие шаги: single-pass online CE, слияние gate+up GEMM, построчное масштабирование
  fp16-сохранений активаций (проверено, но не прошло порог доли ядер в связке).
