"""
step1_crop_and_label.py

кидаешь полные фото накладных в папку raw_invoices/
запускаешь этот скрипт — он открывает каждую накладную,
вырезает зону вагонного номера, показывает тебе кроп,
ты вводишь 7 цифр → файл сохраняется как crops/1616408_001.jpg

запуск:
    python step1_crop_and_label.py

если кроп выглядит плохо (зона съехала) — поправь zone ниже.
"""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import simpledialog, messagebox

# настрой эту зону под свои накладные
# координаты зоны вагонного номера на полном изображении (в пикселях):
# (left, top, right, bottom)
# для твоих smgs накладных — подбери по первой картинке
ZONE = (770, 880, 1130, 1230)

RAW_DIR = Path("raw_invoices")
CROP_DIR = Path("crops")
CROP_DIR.mkdir(exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files():
    files = sorted([f for f in RAW_DIR.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not files:
        print(f"нет изображений в папке {RAW_DIR}/")
        print("положи туда фото накладных и запусти снова.")
        sys.exit(1)
    return files


def already_labelled() -> set:
    """имена исходных файлов, которые уже обработаны."""
    labelled = set()
    for f in CROP_DIR.iterdir():
        if "_" in f.stem:
            original_stem = "_".join(f.stem.split("_")[1:])
            labelled.add(original_stem)
    return labelled


def show_and_label(img_path: Path, crop: Image.Image, index: int, total: int) -> str | None:
    """
    показывает кроп в окне и просит ввести 7 цифр.
    возвращает строку из 7 цифр или none (пропуск).
    """
    w, h = crop.size
    scale = max(1, 400 // h)
    preview = crop.resize((w * scale, h * scale), Image.LANCZOS)

    root = tk.Tk()
    root.title(f"[{index}/{total}] {img_path.name}")
    root.lift()
    root.focus_force()

    from PIL import ImageTk

    tk_img = ImageTk.PhotoImage(preview)
    lbl = tk.Label(root, image=tk_img)
    lbl.pack(padx=10, pady=10)

    info = tk.Label(root, text=f"файл: {img_path.name}", font=("Arial", 10))
    info.pack()

    result = {"value": None}

    def on_submit():
        val = entry.get().strip()
        if len(val) == 7 and val.isdigit():
            result["value"] = val
            root.destroy()
        else:
            tk.messagebox.showerror("ошибка", "введи ровно 7 цифр (0-9)")

    def on_skip():
        root.destroy()

    frame = tk.Frame(root)
    frame.pack(pady=5)

    tk.Label(frame, text="вагонный номер (7 цифр):", font=("Arial", 12)).pack()
    entry = tk.Entry(frame, font=("Arial", 18), width=10, justify="center")
    entry.pack(pady=4)
    entry.focus()
    entry.bind("<Return>", lambda e: on_submit())

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)

    tk.Button(
        btn_frame,
        text="сохранить",
        command=on_submit,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 11),
        width=12,
    ).pack(side="left", padx=5)

    tk.Button(
        btn_frame,
        text="пропустить",
        command=on_skip,
        font=("Arial", 11),
        width=12,
    ).pack(side="left", padx=5)

    root.mainloop()
    return result["value"]


def main():
    files = get_image_files()
    already = already_labelled()

    pending = [f for f in files if f.stem not in already]

    print(f"\nнайдено накладных : {len(files)}")
    print(f"уже размечено     : {len(already)}")
    print(f"осталось разметить: {len(pending)}\n")

    if not pending:
        print("всё уже размечено. переходи к step2.")
        return

    x1, y1, x2, y2 = ZONE
    saved = 0

    for i, fpath in enumerate(pending, 1):
        try:
            img = Image.open(fpath)
        except Exception as e:
            print(f"не могу открыть {fpath.name}: {e}")
            continue

        iw, ih = img.size
        print(f"[{i}/{len(pending)}] {fpath.name} ({iw}x{ih}px)")

        crop = img.crop((x1, y1, x2, y2)).convert("L")

        label = show_and_label(fpath, crop, i, len(pending))

        if label is None:
            print("пропущено")
            continue

        out_name = f"{label}_{fpath.stem}.jpg"
        out_path = CROP_DIR / out_name
        crop.save(out_path, "JPEG", quality=95)

        print(f"сохранён -> crops/{out_name}")
        saved += 1

    print(f"\nготово. сохранено {saved} кропов в crops/")
    print("следующий шаг: python step2_split.py")


if __name__ == "__main__":
    main()
