"""
main.py

Standalone CLI entry point. Preserves the original script's usage
pattern - run locally, capture a list of students one after another -
while using the new reusable DatasetCollector service underneath.

This is just ONE consumer of the service. A FastAPI/Flask backend
would import `DatasetCollector` and `StudentDetails` directly instead
of running this file (see README.md for an example).

Run with:  python -m face_dataset_service.main
"""

import logging
import time

try:
    from .dataset_collector import DatasetCollector
    from .models import StudentDetails
except ImportError:  # Allow running the file directly as a script
    from dataset_collector import DatasetCollector
    from models import StudentDetails

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def cli_progress_printer(student, saved, target, message):
    """Simple console progress line. A backend would replace this with a websocket/SSE push."""
    print(f"\r{student.name} - {saved}/{target}  ({message})".ljust(60), end="", flush=True)


def main():
    # In production this list will come from a database, populated by
    # an admin through the frontend dashboard. It is kept here only to
    # preserve the original script's "run for a list of students" flow.
    students = [
        StudentDetails(name="Nipun", roll_number="101", department="CSE"),
        StudentDetails(name="Yug", roll_number="102", department="CSE"),
        StudentDetails(name="Rajkumari", roll_number="103", department="CSE"),
        
    ]

    collector = DatasetCollector()

    for student in students:
        print(f"\nGet ready: {student.name}. Starting in 5 seconds...")
        time.sleep(5)

        result = collector.collect(
            student=student,
            show_preview=True,
            progress_callback=cli_progress_printer,
        )

        print(f"\n{result.message} Saved {result.images_saved}/{result.target_images} "
              f"images to {result.folder_path}")


if __name__ == "__main__":
    main()