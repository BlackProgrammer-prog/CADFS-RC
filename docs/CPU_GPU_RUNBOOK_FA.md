# راهنمای اجرای مرحله‌به‌مرحله CPU و GPU

این راهنما non-destructive است: فایل‌های فعلی `results/models/tuned_next.json`
و `results/logs/paper_fast.csv` نباید overwrite شوند. تمام اجراهای جدید در
پوشه‌های نسخه‌دار مانند `cpu_v1` و `gpu_v1` ذخیره می‌شوند.

## 1. قانون علمی قبل از اجرا

- توسعه و تیون فقط روی `val`، `val_structural` و `val_shift` انجام شود.
- testهای فعلی قبلاً دیده شده‌اند. بعد از هر اصلاح جدید، برای نتیجه نهایی
  مقاله باید `final_test` و OODهای نهایی تازه ساخته و تا زمان freeze باز نشوند.
- شعاع ناحیه‌ای صفر رفتار دقیق فعلی را حفظ می‌کند. شعاع ۱ پروفایل efficiency
  پیشنهادی است؛ شعاع‌های ۳ و ۷ فقط ablation هستند.

## 2. نصب مشترک

همه دستورها از ریشه مخزن در WSL/Linux اجرا می‌شوند.

```bash
cd ~/mind/CADFS-RC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU:

```bash
python -m pip install -r requirements-cpu.txt
```

سرور NVIDIA:

```bash
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

خروجی فرمان دوم باید `True` و نام GPU را نشان دهد.

## 3. build نسخه Release

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release -DCADFS_NATIVE_ARCH=ON -DCADFS_ENABLE_IPO=ON -DCMAKE_TOOLCHAIN_FILE="$PWD/.vcpkg/scripts/buildsystems/vcpkg.cmake" -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build cmake-build-release -j "$(nproc)"
ctest --test-dir cmake-build-release --output-on-failure
```

اگر binary باید بین CPUهای مختلف جابه‌جا شود، `CADFS_NATIVE_ARCH=OFF` قرار
دهید. benchmark نهایی را با binary ساخته‌شده روی همان دستگاه مقصد اجرا کنید.

## 4. بررسی داده و label

```bash
test -f data/instances/train.csv
test -f data/labels/train.npz
test -f data/labels/val_shift.npz
```

اگر labelها وجود ندارند:

```bash
python python/scripts/make_labels.py --splits train train_structural val val_structural val_shift --samples-per-goal 300 --seed 7 --workers "$(nproc)"
```

## 5. مسیر CPU-base

### 5.1 آموزش student نسخه‌دار

```bash
mkdir -p results/models/cpu_v1
python python/ml/train_student.py --device cpu --epochs 50 --batch-size 512 --supervised-weight 0.80 --rank-weight 0.20 --teacher-weight 0 --seed 2026 --artifacts-dir results/models/cpu_v1 --out results/models/cpu_v1/fast_ensemble.txt
```

خروجی‌ها:

```text
results/models/cpu_v1/fast_student.pt
results/models/cpu_v1/fast_ensemble.txt
results/models/cpu_v1/calibration_fast.json
results/models/cpu_v1/training_manifest_fast.json
```

### 5.2 بررسی parity

```bash
python python/ml/check_cpp_parity.py --backend fast --model results/models/cpu_v1/fast_ensemble.txt --checkpoints-dir results/models/cpu_v1 --split val_structural --samples 64
```

### 5.3 تیون baseline دقیق فعلی

```bash
python python/scripts/tune_next_validation.py --splits val val_structural val_shift --per-split 100 --workers 1 --seed 17 --guidance fast --guidance-model results/models/cpu_v1/fast_ensemble.txt --guidance-region-radius 0 --out results/models/cpu_v1/tuned_next.json --out-conservative results/models/cpu_v1/tuned_next_conservative.json
```

### 5.4 تیون metric-region شعاع ۱

شعاع ۱ یک cell محلی `3x3` می‌سازد. مدل در اولین node هر cell اجرا می‌شود و
residual آموخته‌شده نسبت به anchor برای nodeهای همان cell reuse می‌شود.
این قابلیت فقط ترتیب ثانویه داخل FOCAL را عوض می‌کند و proof کران را تغییر
نمی‌دهد.

```bash
python python/scripts/tune_next_validation.py --splits val val_structural val_shift --per-split 100 --workers 1 --seed 17 --guidance fast --guidance-model results/models/cpu_v1/fast_ensemble.txt --guidance-region-radius 1 --out results/models/cpu_v1/tuned_next_metric_r1.json --out-conservative results/models/cpu_v1/tuned_next_metric_r1_conservative.json
```

### 5.5 smoke test

```bash
python python/scripts/smoke_next.py --split val_shift --instances 10 --guidance fast --guidance-model results/models/cpu_v1/fast_ensemble.txt --tuned-next results/models/cpu_v1/tuned_next.json --methods astar wastar cadfs_next cadfs_next_metric_r1
```

## 6. مسیر GPU-base

GPU فعلاً برای آموزش teacher/student استفاده می‌شود. search نهایی C++/CPU
است؛ تا زمان پیاده‌سازی batch FOCAL نباید ادعای `GPU-accelerated search`
مطرح شود.

### 6.1 آموزش و export کردن teacher نسخه‌دار

```bash
mkdir -p results/models/gpu_v1
python python/ml/train_ensemble.py --device cuda --amp --compile --K 7 --epochs 80 --batch-size 2048 --lr 3e-4 --patience 10 --structural-weight 1.0 --hidden 96 --artifacts-dir results/models/gpu_v1
python python/ml/export_weights.py --models-dir results/models/gpu_v1 --out results/models/gpu_v1/ensemble.txt
```

### 6.2 آموزش student با distillation

```bash
python python/ml/train_student.py --device cuda --amp --compile --epochs 50 --batch-size 2048 --supervised-weight 0.60 --rank-weight 0.15 --teacher-weight 0.25 --teacher-dir results/models/gpu_v1 --seed 2026 --artifacts-dir results/models/gpu_v1 --out results/models/gpu_v1/fast_ensemble.txt
```

در صورت کمبود VRAM، batch را ابتدا 1024 و سپس 512 قرار دهید.

### 6.3 parity و تیون روی CPU مقصد

```bash
python python/ml/check_cpp_parity.py --backend fast --model results/models/gpu_v1/fast_ensemble.txt --checkpoints-dir results/models/gpu_v1 --split val_structural --samples 64
python python/scripts/tune_next_validation.py --splits val val_structural val_shift --per-split 100 --workers 1 --seed 17 --guidance fast --guidance-model results/models/gpu_v1/fast_ensemble.txt --guidance-region-radius 1 --out results/models/gpu_v1/tuned_next_metric_r1.json --out-conservative results/models/gpu_v1/tuned_next_metric_r1_conservative.json
```

latency tuning و benchmark باید روی CPU مقصد و با یک worker انجام شوند، حتی
اگر مدل روی GPU آموزش دیده باشد.

## 7. benchmark توسعه روی validation

```bash
python python/scripts/run_experiments.py --splits val val_structural val_shift --per-split 100 --methods dijkstra astar wastar focal_plain cadfs_next cadfs_next_metric_r1 cadfs_next_metric_r3 cadfs_next_metric_r7 --guidance fast --guidance-model results/models/cpu_v1/fast_ensemble.txt --tuned-next results/models/cpu_v1/tuned_next.json --warmup-runs 1 --repetitions 3 --method-order rotate --out results/logs/metric_validation_cpu_v1.csv
python python/analysis/tables.py --input results/logs/metric_validation_cpu_v1.csv --tag metric_validation_cpu_v1
python python/analysis/systems_table.py --input results/logs/metric_validation_cpu_v1.csv --tag metric_validation_cpu_v1 --baseline astar
```

Runner فایل موجود را بدون `--overwrite` بازنویسی نمی‌کند.

## 8. benchmark نهایی مقاله

این مرحله فقط پس از freeze و روی splitهای final دست‌نخورده اجرا شود:

```bash
taskset -c 2 python python/scripts/run_experiments.py --splits final_test final_shift_density final_shift_size final_shift_family --per-split 200 --methods dijkstra astar wastar focal_plain learn_focal_wstar cadfs cadfs_next cadfs_next_metric_tuned --guidance fast --guidance-model results/models/cpu_v1/fast_ensemble.txt --tuned-next results/models/cpu_v1/tuned_next.json --tuned-next-metric results/models/cpu_v1/tuned_next_metric_r1.json --warmup-runs 2 --repetitions 5 --method-order rotate --out results/logs/final_cpu_v1.csv
python python/analysis/tables.py --input results/logs/final_cpu_v1.csv --tag final_cpu_v1
python python/analysis/systems_table.py --input results/logs/final_cpu_v1.csv --tag final_cpu_v1 --baseline astar
python python/analysis/figures_next.py --input results/logs/final_cpu_v1.csv --tag final_cpu_v1 --bound 2.0
```

برای timing، performance governor را در صورت دسترسی روی `performance` قرار
دهید و workload دیگری روی دستگاه اجرا نکنید.

## 9. قفل artifacts

```bash
sha256sum results/models/cpu_v1/fast_student.pt results/models/cpu_v1/fast_ensemble.txt results/models/cpu_v1/tuned_next_metric_r1.json results/logs/final_cpu_v1.csv results/logs/final_cpu_v1.manifest.json
git rev-parse HEAD
git status --short
```

Runner hash مدل، tuning JSON، CSV splitها، commit، وضعیت dirty، packageها،
CPU، warm-up، repetitions و ترتیب روش‌ها را در manifest ثبت می‌کند.

## 10. gate پذیرش روش metric

روش metric فقط وقتی جایگزین baseline شود که روی final test:

- success و bound حفظ شوند؛
- quality gate از پیش تعیین‌شده، مثلاً `max ratio <= 1.30`، رعایت شود؛
- کاهش runtime پس از Holm correction معنی‌دار باشد؛
- trade-off expansion از قبل تعریف و کامل گزارش شود؛
- هیچ split شکست پنهان‌شده‌ای نداشته باشد.

در غیر این صورت baseline شعاع صفر نتیجه اصلی می‌ماند و metric-region فقط
به‌عنوان ablation یا crossover گزارش می‌شود.

## 11. نتیجه مهندسی اولیه metric-region

روی ۳۰۰ مسئله validation با مدل فعلی و بدون تیون مجدد شعاع:

| روش | زمان میانگین | ارزیابی مدل | ratio میانگین | expansion |
|---|---:|---:|---:|---:|
| radius 0 | 72.19 ms | 2784.7 | 1.0629 | 2734.7 |
| radius 1 | 13.78 ms | 436.6 | 1.0603 | 2756.5 |
| radius 3 | 5.30 ms | 96.5 | 1.0588 | 2872.5 |
| radius 7 | 3.96 ms | 28.6 | 1.0408 | 3136.8 |

شعاع ۱ حدود ۵.۲ برابر سریع‌تر شد، ratio میانگین اندکی بهتر شد و expansion
حدود ۰.۸٪ افزایش یافت. این فقط pilot validation است و نتیجه test یا ادعای
مقاله محسوب نمی‌شود.

## 12. تکرار seedهای آموزش

برای مقاله student را حداقل با seedهای 2026، 2027 و 2028 در سه پوشه جدا
آموزش دهید. انتخاب architecture با validation انجام شود و نتیجه نهایی علاوه
بر bootstrap queryها، variation میان seedهای آموزش را نیز گزارش کند.
