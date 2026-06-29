"""
calibrate_zone.py
─────────────────
зажми мышку и нарисуй прямоугольник вокруг ячейки с номером вагона.
отпусти — получишь zone = (x1, y1, x2, y2).

запуск:
    python calibrate_zone.py
"""

import sys
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk

RAW_DIR = Path("raw_invoices")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
DISPLAY_SCALE = 0.6


def main():
    files = sorted(f for f in RAW_DIR.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not files:
        print(f"нет файлов в {RAW_DIR}/")
        sys.exit(1)

    img_path = files[0]
    print(f"открываю: {img_path.name}")

    orig = Image.open(img_path).convert("RGB")
    ow, oh = orig.size
    dw, dh = int(ow * DISPLAY_SCALE), int(oh * DISPLAY_SCALE)
    display = orig.resize((dw, dh), Image.LANCZOS)

    root = tk.Tk()
    root.title(f"зажми и нарисуй прямоугольник вокруг номера вагона — {img_path.name}")

    info = tk.Label(
        root,
        text="зажми левую кнопку мыши и нарисуй прямоугольник вокруг ячейки с номером вагона",
        font=("Arial", 11),
        fg="#1a73e8",
        wraplength=dw,
    )
    info.pack(pady=6)

    canvas = tk.Canvas(root, width=dw, height=dh, cursor="crosshair")
    canvas.pack()

    tk_img = ImageTk.PhotoImage(display)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    state = {"start": None, "rect": None}

    def on_press(event):
        state["start"] = (event.x, event.y)
        if state["rect"]:
            canvas.delete(state["rect"])
            state["rect"] = None

    def on_drag(event):
        if not state["start"]:
            return

        x0, y0 = state["start"]

        if state["rect"]:
            canvas.delete(state["rect"])

        state["rect"] = canvas.create_rectangle(
            x0,
            y0,
            event.x,
            event.y,
            outline="red",
            width=2,
        )

    def on_release(event):
        if not state["start"]:
            return

        x0d, y0d = state["start"]
        x1d, y1d = event.x, event.y

        lx, rx = sorted([x0d, x1d])
        ty, by = sorted([y0d, y1d])

        if rx - lx < 5 or by - ty < 5:
            info.config(
                text="слишком маленький прямоугольник, попробуй ещё раз",
                fg="red",
            )
            state["start"] = None
            return

        x1 = int(lx / DISPLAY_SCALE)
        y1 = int(ty / DISPLAY_SCALE)
        x2 = int(rx / DISPLAY_SCALE)
        y2 = int(by / DISPLAY_SCALE)

        zone_str = f"zone = ({x1}, {y1}, {x2}, {y2})"

        print(f"\nтвоя зона:\n   {zone_str}")
        print("скопируй эту строку в step1_crop_and_label.py\n")

        info.config(
            text=f"zone = ({x1}, {y1}, {x2}, {y2}) — скопируй в step1_crop_and_label.py",
            fg="#2e7d32",
        )

        crop = orig.crop((x1, y1, x2, y2))
        if crop.width > 0 and crop.height > 0:
            crop.save("calibration_preview.jpg")
            print("превью -> calibration_preview.jpg")

        state["start"] = None

        tk.Button(
            root,
            text="нарисовать заново",
            command=lambda: info.config(
                text="зажми снова и нарисуй новый прямоугольник",
                fg="#1a73e8",
            ),
            font=("Arial", 10),
        ).pack(side="left", padx=10, pady=6)

        tk.Button(
            root,
            text="закрыть",
            command=root.destroy,
            font=("Arial", 10),
        ).pack(side="right", padx=10, pady=6)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    root.mainloop()


if __name__ == "__main__":
    main()
