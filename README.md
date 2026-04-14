# YOLOv8 TensorRT Optimization & Benchmarking
> Исследование производительности компьютерного зрения: от стандартного PyTorch до кастомных движков TensorRT.

## Project Overview
Цель проекта — оптимизировать детекцию YOLOv8 для задач реального времени. В ходе работы реализован полный цикл конвертации модели и проведено профилирование метрик на различных бэкендах.

## 🛠 Engineering Highlights
*   **Model Transformation:** Оптимизация графа вычислений через цепочку **PyTorch → ONNX → TensorRT**.
*   **Buffer Management:** Прямое управление памятью GPU с помощью **PyCUDA**: аллокация буферов, асинхронная передача данных (Host-to-Device/Device-to-Host).
*   **Quantization Impact:** Сравнительный анализ точности и скорости при переходе с FP32 на FP16.
*   **Accurate Benchmarking:** Замеры latency с учетом прогрева (warm-up) и исключения оверхеда на I/O.

## Performance Benchmarks (Tesla T4)


| Backend | Device | Latency | Speedup | Max Difference |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | CPU | 86.53 ms | 1.0x | — |
| **ONNX** | CPU | 64.32 ms | 1.34x | 5e-4 |
| **TensorRT (FP32)** | **GPU** | **43.26 ms** | **2.0x** | 6e-4 |
| **TensorRT (FP16)** | **GPU** | **39.47 ms** | **2.2x** | 3.07 |

> **Summary:** Перенос инференса на TensorRT ускорил обработку в 2 раза. Использование FP16 дает дополнительный прирост, но требует контроля допустимых отклонений (max_diff).

## Tech Stack
`Python` • `PyTorch` • `TensorRT` • `ONNX` • `PyCUDA` • `YOLOv8`

---
*Сфокусировано на практическом применении ML System Engineering и оптимизации инференса.*
