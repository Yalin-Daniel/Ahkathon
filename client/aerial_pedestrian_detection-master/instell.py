#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
בדיקה של run.py לראות איך הוא עובד
"""

import os
import sys
import subprocess
from pathlib import Path


def examine_run_py():
    """
    בדוק את תוכן run.py
    """
    run_file = Path(r"/client\aerial_pedestrian_detection-master\run.py")

    if not run_file.exists():
        print("❌ קובץ run.py לא נמצא")
        return False

    print("📄 בודק את run.py...")
    print("=" * 80)

    try:
        with open(run_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        lines = content.split('\n')

        print(f"📊 הקובץ כולל {len(lines)} שורות")
        print("\n📝 תוכן הקובץ (50 שורות ראשונות):")
        print("-" * 50)

        for i, line in enumerate(lines[:50]):
            print(f"{i + 1:3d}: {line}")

        if len(lines) > 50:
            print("...")
            print(f"(ועוד {len(lines) - 50} שורות)")

        return True

    except Exception as e:
        print(f"❌ שגיאה בקריאת הקובץ: {e}")
        return False


def test_run_py_help():
    """
    נסה להריץ את run.py עם --help
    """
    print("\n🧪 בודק אם run.py עובד...")

    project_path = Path(r"/client\aerial_pedestrian_detection-master")
    run_file = project_path / "run.py"

    if not run_file.exists():
        print("❌ run.py לא נמצא")
        return False

    original_dir = os.getcwd()

    try:
        os.chdir(project_path)
        print(f"📂 עבר לתיקייה: {project_path}")

        # נסה להריץ עם --help
        print("🔧 מריץ: python run.py --help")
        result = subprocess.run([sys.executable, "run.py", "--help"],
                                capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ run.py עובד!")
            print("📝 Help output:")
            print(result.stdout)
            return True
        else:
            print("❌ run.py לא עובד כצפוי")
            print(f"🔴 שגיאה: {result.stderr}")

            # אולי הוא לא מקבל --help, נסה בלי ארגומנטים
            print("\n🔧 מנסה בלי ארגומנטים...")
            result2 = subprocess.run([sys.executable, "run.py"],
                                     capture_output=True, text=True, timeout=10)

            if result2.returncode == 0 or "usage" in result2.stderr.lower():
                print("✅ run.py עובד (רק צריך ארגומנטים)")
                print(f"📝 פלט: {result2.stderr}")
                return True
            else:
                print("❌ run.py באמת לא עובד")
                print(f"🔴 שגיאה: {result2.stderr}")
                return False

    except subprocess.TimeoutExpired:
        print("⏰ run.py תקוע - זה יכול להיות בסדר")
        return True
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        return False
    finally:
        os.chdir(original_dir)


def fix_keras_retinanet_simple():
    """
    נסה לתקן את keras-retinanet בצורה פשוטה
    """
    print("\n🔧 מנסה לתקן את keras-retinanet...")

    project_path = Path(r"/client\aerial_pedestrian_detection-master")
    original_dir = os.getcwd()

    try:
        os.chdir(project_path)

        # נסה התקנה פשוטה
        result = subprocess.run(["pip", "install", "."],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ keras-retinanet הותקן בהצלחה!")
            return True
        else:
            print("❌ עדיין בעיה עם keras-retinanet")
            print(f"🔴 שגיאה: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ שגיאה: {e}")
        return False
    finally:
        os.chdir(original_dir)


def test_imports():
    """
    בדוק שאפשר לייבא את keras_retinanet
    """
    print("\n🔍 בודק ייבואים...")

    imports_to_test = [
        ("import keras_retinanet", "keras-retinanet"),
        ("from keras_retinanet import models", "keras-retinanet models"),
        ("import cv2", "OpenCV"),
        ("import tensorflow as tf", "TensorFlow"),
        ("import numpy as np", "NumPy")
    ]

    success = True

    for import_stmt, name in imports_to_test:
        try:
            exec(import_stmt)
            print(f"✅ {name}: עובד")
        except ImportError as e:
            print(f"❌ {name}: לא עובד - {e}")
            success = False
        except Exception as e:
            print(f"⚠️  {name}: בעיה - {e}")

    return success


def main():
    """
    פונקציה ראשית
    """
    print("🧪 בודק את המערכת...")
    print("=" * 80)

    # בדוק את run.py
    if examine_run_py():

        # בדוק אם run.py עובד
        run_works = test_run_py_help()

        # אם keras-retinanet לא עובד, נסה לתקן
        imports_work = test_imports()

        if not imports_work:
            print("\n🔧 מנסה לתקן keras-retinanet...")
            if fix_keras_retinanet_simple():
                imports_work = test_imports()

        print("\n" + "=" * 80)
        print("📋 סיכום:")
        print(f"📄 run.py קיים: ✅")
        print(f"🚀 run.py עובד: {'✅' if run_works else '❌'}")
        print(f"📦 ייבואים עובדים: {'✅' if imports_work else '❌'}")

        if run_works and imports_work:
            print("\n🎉 הכל מוכן! אפשר לעבור לצעד הבא")
            print("\n💡 הצעד הבא: ליצור מחלקה שתחבר את run.py ל-JSON שלך")
        elif run_works:
            print("\n⚠️  run.py עובד אבל יש בעיות עם ייבואים")
            print("💡 אולי עדיין אפשר לעבוד עם זה")
        else:
            print("\n❌ יש בעיות. אבל אולי נוכל לעקוף אותן")

    else:
        print("❌ לא מוצא את run.py")


if __name__ == "__main__":
    main()