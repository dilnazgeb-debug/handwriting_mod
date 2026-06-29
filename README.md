# handwriting_mod
шаг 0 — найти координаты зоны (один раз)
bash# кинь одну накладную в raw_invoices/
python calibrate_zone.py
# мышкой кликаешь  зоны с номером вагона
# получаешь ZONE = (x1, y1, x2, y2) → вставляешь в step1
шаг 1 — разметить все накладные
bash# Кидаешь ВСЕ накладные в raw_invoices/
python step1_crop_and_label.py
# открывается окно с кропом → вводишь 7 цифр → Enter → следующая
# файлы сохраняются как: crops/1616408_invoice001.jpg
Шаг 2–4 — сплит, обучение, оценка
bashpython step2_split.py
python train.py
python evaluate.py --checkpoint checkpoints/best_model.pt
