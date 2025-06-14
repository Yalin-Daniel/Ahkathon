import os
from pathlib import Path


def check_project_structure():
    """
    בדוק את המבנה של המאגר - גרסה שעובדת מכל מקום
    """
    print("🔍 מחפש את התיקייה aerial_pedestrian_detection-master...")

    # הנתיב הספציפי שציינת
    specific_path = Path(r"/client\aerial_pedestrian_detection-master")

    if specific_path.exists():
        pedestrian_dir = specific_path
        print(f"✅ נמצאת התיקייה ב: {pedestrian_dir}")
    else:
        print("❌ התיקייה לא נמצאת במיקום הצפוי")
        print("🔍 מחפש במקומות אחרים...")

        # חפש בכל הדיסק C (רק בתיקיות נפוצות)
        search_paths = [
            Path.cwd(),
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path("C:/"),
        ]

        found = False
        for search_path in search_paths:
            if search_path.exists():
                matches = list(search_path.rglob("aerial_pedestrian_detection-master"))
                if matches:
                    pedestrian_dir = matches[0]
                    print(f"✅ נמצא ב: {pedestrian_dir}")
                    found = True
                    break

        if not found:
            print("❌ לא נמצאת התיקייה בשום מקום")
            return False

    # רשום את כל הקבצים והתיקיות
    all_items = list(pedestrian_dir.rglob("*"))

    print(f"\n📁 נמצאו {len(all_items)} פריטים סה\"כ")

    # קבצי Python
    py_files = [f for f in all_items if f.suffix == '.py']
    print(f"\n🐍 קבצי Python ({len(py_files)}):")
    for py_file in sorted(py_files)[:15]:  # הראה 15 הראשונים
        rel_path = py_file.relative_to(pedestrian_dir)
        print(f"  - {rel_path}")
    if len(py_files) > 15:
        print(f"  ... ועוד {len(py_files) - 15} קבצים")

    # קובץ README
    readme_files = list(pedestrian_dir.rglob("README*"))
    if readme_files:
        print(f"\n📖 README files:")
        for readme in readme_files:
            rel_path = readme.relative_to(pedestrian_dir)
            print(f"  - {rel_path}")

    # requirements
    req_files = list(pedestrian_dir.rglob("requirements*"))
    if req_files:
        print(f"\n📋 Requirements files:")
        for req in req_files:
            rel_path = req.relative_to(pedestrian_dir)
            print(f"  - {rel_path}")

    # קבצי מודל
    model_extensions = ['.h5', '.hdf5', '.pb', '.pth', '.pt', '.weights', '.onnx']
    model_files = []
    for ext in model_extensions:
        model_files.extend(list(pedestrian_dir.rglob(f"*{ext}")))

    if model_files:
        print(f"\n🤖 קבצי מודל ({len(model_files)}):")
        for model in sorted(model_files):
            rel_path = model.relative_to(pedestrian_dir)
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  - {rel_path} ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠️  לא נמצאו קבצי מודל מאומנים")

    # תיקיות עיקריות
    main_dirs = [d for d in pedestrian_dir.iterdir() if d.is_dir()]
    if main_dirs:
        print(f"\n📂 תיקיות עיקריות:")
        for dir in sorted(main_dirs):
            file_count = len(list(dir.rglob("*")))
            print(f"  - {dir.name}/ ({file_count} פריטים)")

    # חפש קבצים חשובים
    important_files = ['main.py', 'run.py', 'detect.py', 'inference.py', 'demo.py', 'test.py']
    found_important = []

    for imp_file in important_files:
        matches = list(pedestrian_dir.rglob(imp_file))
        if matches:
            found_important.extend(matches)

    if found_important:
        print(f"\n🎯 קבצים חשובים שנמצאו:")
        for imp in found_important:
            rel_path = imp.relative_to(pedestrian_dir)
            print(f"  - {rel_path}")

    # שמור את הנתיב למשתנה גלובלי לשימוש בפונקציות אחרות
    globals()['pedestrian_dir'] = pedestrian_dir
    return True


def read_readme():
    """
    קרא את ה-README אם קיים
    """
    if 'pedestrian_dir' not in globals():
        print("❌ תחילה הרץ את check_project_structure()")
        return

    pedestrian_dir = globals()['pedestrian_dir']
    readme_files = list(pedestrian_dir.rglob("README*"))

    if readme_files:
        readme_file = readme_files[0]  # קח את הראשון
        print(f"\n📖 קורא {readme_file.name}:")
        print("=" * 80)

        try:
            with open(readme_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # הראה חלק מהתוכן
            lines = content.split('\n')
            for i, line in enumerate(lines[:40]):  # 40 שורות ראשונות
                print(f"{i + 1:2d}: {line}")

            if len(lines) > 40:
                print("...")
                print(f"(ועוד {len(lines) - 40} שורות)")

        except Exception as e:
            print(f"❌ שגיאה בקריאת הקובץ: {e}")
    else:
        print("\n⚠️  לא נמצא קובץ README")


def check_requirements():
    """
    בדוק requirements
    """
    if 'pedestrian_dir' not in globals():
        print("❌ תחילה הרץ את check_project_structure()")
        return

    pedestrian_dir = globals()['pedestrian_dir']
    req_files = list(pedestrian_dir.rglob("requirements*"))

    if req_files:
        req_file = req_files[0]
        print(f"\n📋 תוכן {req_file.name}:")
        print("-" * 50)

        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                requirements = f.read().strip()
            print(requirements)

            # הצע התקנה
            print(f"\n💡 להתקין את החבילות הללו, הריצי:")
            print(f"pip install -r \"{req_file}\"")

        except Exception as e:
            print(f"❌ שגיאה בקריאת requirements: {e}")
    else:
        print("\n⚠️  לא נמצא קובץ requirements")


if __name__ == "__main__":
    print("🚁 בודק את המאגר aerial_pedestrian_detection...")
    print("=" * 80)

    if check_project_structure():
        read_readme()
        check_requirements()

        print("\n" + "=" * 80)
        print("✅ בדיקה הושלמה!")
        print(f"\nהתיקייה נמצאת ב: {globals().get('pedestrian_dir', 'לא נמצא')}")
        print("\n💡 הצעדים הבאים:")
        print("1. קרא את ה-README בעיון")
        print("2. התקן את החבילות מ-requirements")
        print("3. בדוק איך לרוץ את הקוד")
    else:
        print("❌ לא ניתן למצוא את התיקייה")