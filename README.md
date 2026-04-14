# High-Performance YOLOv8: TensorRT Optimization & Benchmarking

Реализация пайплайна оптимизации компьютерного зрения для задач реального времени. Переход от стандартного PyTorch к высокопроизводительным движкам с анализом системных метрик.

## Computer Science Highlights
Проект демонстрирует навыки работы с низкоуровневой оптимизацией GPU:
*   **Engine Serialization:** Конвертация статического графа PyTorch → ONNX → TensorRT.
*   **Memory Management:** Прямое управление аллокацией видеопамяти через **PyCUDA** (async memcpy, host-to-device).
*   **Precision Engineering:** Исследование влияния разрядности (FP32 vs FP16) на пропускную способность и накопление численной ошибки.
*   **Profiling:** Кастомный бенчмаркинг с учетом "warm-up" итераций для точного замера latency.

## Performance Analysis (Tesla T4)


| Format | Runtime | Latency (ms) | Speedup | Max Diff (vs PT) |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | CPU | 86.53 | 1.0x | - |
| **ONNX** | CPU | 64.32 | 1.34x | 0.0005 |
| **TensorRT FP32**| **GPU** | **43.26** | **2.0x** | 0.0006 |
| **TensorRT FP16**| **GPU** | **39.47** | **2.19x** | 3.0718 |

**Key Insight:** Оптимизация через TensorRT на GPU дала двукратный прирост скорости. Переход на FP16 дает дополнительный буст, но требует контроля точности из-за роста `max_diff`.

## Tech Stack
`Python` • `PyTorch` • `TensorRT` • `ONNX` • `PyCUDA` • `YOLOv8`

---
*Developed as a practical study in ML System Engineering.*
