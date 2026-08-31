"""
Unit tests for backend Pydantic models.

Covers:
  - StudentCreate validation (roll_number pattern, max_length, required fields)
  - StudentUpdate optional fields
  - StudentResponse structure
  - SettingsUpdate field bounds (ge/le constraints)
  - SettingsResponse shape
  - AttendanceRecord / AttendanceCreate / ManualAttendance
  - FrameRequest / FrameResult / RecognitionResponse
  - DashboardStats / TrendData / DepartmentStats
"""
import pytest
from pydantic import ValidationError

from backend.models.student import StudentCreate, StudentUpdate, StudentResponse
from backend.models.settings import SettingsUpdate, SettingsResponse
from backend.models.attendance import AttendanceRecord, AttendanceCreate, ManualAttendance
from backend.models.recognition import FrameRequest, FrameResult, RecognitionResponse
from backend.models.analytics import DashboardStats, TrendData, DepartmentStats


# ---------------------------------------------------------------------------
# StudentCreate
# ---------------------------------------------------------------------------

class TestStudentCreate:
    def test_valid_minimal(self):
        s = StudentCreate(roll_number="101", name="Alice")
        assert s.roll_number == "101"
        assert s.name == "Alice"
        assert s.department == ""
        assert s.email == ""
        assert s.phone == ""

    def test_valid_full(self):
        s = StudentCreate(
            roll_number="A1b2C3",
            name="Bob",
            department="CS",
            email="bob@example.com",
            phone="9999999999",
        )
        assert s.roll_number == "A1b2C3"

    def test_roll_number_pattern_rejects_special_chars(self):
        with pytest.raises(ValidationError):
            StudentCreate(roll_number="101-A", name="Alice")

    def test_roll_number_pattern_rejects_spaces(self):
        with pytest.raises(ValidationError):
            StudentCreate(roll_number="101 A", name="Alice")

    def test_roll_number_max_length(self):
        with pytest.raises(ValidationError):
            StudentCreate(roll_number="A" * 21, name="Alice")

    def test_roll_number_exactly_20_chars_ok(self):
        s = StudentCreate(roll_number="A" * 20, name="Alice")
        assert len(s.roll_number) == 20

    def test_roll_number_empty_raises(self):
        with pytest.raises(ValidationError):
            StudentCreate(roll_number="", name="Alice")

    def test_name_empty_raises(self):
        with pytest.raises(ValidationError):
            StudentCreate(roll_number="101", name="")

    def test_name_max_length(self):
        with pytest.raises(ValidationError):
            StudentCreate(roll_number="101", name="A" * 101)

    def test_name_exactly_100_chars_ok(self):
        s = StudentCreate(roll_number="101", name="A" * 100)
        assert len(s.name) == 100


# ---------------------------------------------------------------------------
# StudentUpdate
# ---------------------------------------------------------------------------

class TestStudentUpdate:
    def test_all_none_is_valid(self):
        u = StudentUpdate()
        assert u.name is None
        assert u.department is None
        assert u.email is None
        assert u.phone is None

    def test_partial_update_valid(self):
        u = StudentUpdate(name="Charlie", department="Math")
        assert u.name == "Charlie"
        assert u.email is None

    def test_name_empty_raises(self):
        with pytest.raises(ValidationError):
            StudentUpdate(name="")


# ---------------------------------------------------------------------------
# SettingsUpdate
# ---------------------------------------------------------------------------

class TestSettingsUpdate:
    def test_all_none_valid(self):
        s = SettingsUpdate()
        assert s.recognition_threshold is None

    def test_recognition_threshold_in_range(self):
        s = SettingsUpdate(recognition_threshold=0.5)
        assert s.recognition_threshold == 0.5

    def test_recognition_threshold_below_zero_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(recognition_threshold=-0.01)

    def test_recognition_threshold_above_one_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(recognition_threshold=1.01)

    def test_recognition_threshold_boundary_values(self):
        assert SettingsUpdate(recognition_threshold=0.0).recognition_threshold == 0.0
        assert SettingsUpdate(recognition_threshold=1.0).recognition_threshold == 1.0

    def test_cooldown_period_below_zero_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(cooldown_period_seconds=-1)

    def test_cooldown_period_above_max_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(cooldown_period_seconds=86401)

    def test_cooldown_period_boundary_values(self):
        assert SettingsUpdate(cooldown_period_seconds=0).cooldown_period_seconds == 0
        assert SettingsUpdate(cooldown_period_seconds=86400).cooldown_period_seconds == 86400

    def test_stable_frame_count_below_one_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(stable_frame_count=0)

    def test_stable_frame_count_above_30_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(stable_frame_count=31)

    def test_camera_index_below_zero_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(camera_index=-1)

    def test_camera_index_above_9_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(camera_index=10)

    def test_blur_threshold_below_zero_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(blur_threshold=-0.1)

    def test_blur_threshold_above_500_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(blur_threshold=500.1)

    def test_min_face_size_below_10_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(min_face_size=9)

    def test_min_face_size_above_500_raises(self):
        with pytest.raises(ValidationError):
            SettingsUpdate(min_face_size=501)

    def test_partial_update_only_specified_keys(self):
        s = SettingsUpdate(recognition_threshold=0.7)
        assert s.recognition_threshold == 0.7
        assert s.cooldown_period_seconds is None


# ---------------------------------------------------------------------------
# SettingsResponse
# ---------------------------------------------------------------------------

class TestSettingsResponse:
    def test_construct_valid(self):
        r = SettingsResponse(
            recognition_threshold=0.6,
            cooldown_period_seconds=300,
            stable_frame_count=4,
            camera_index=0,
            blur_threshold=50.0,
            min_face_size=60,
        )
        assert r.recognition_threshold == 0.6
        assert r.min_face_size == 60


# ---------------------------------------------------------------------------
# AttendanceRecord / AttendanceCreate / ManualAttendance
# ---------------------------------------------------------------------------

class TestAttendanceModels:
    def test_attendance_record_valid(self):
        r = AttendanceRecord(
            id=1,
            roll_number="101",
            name="Alice",
            date="2024-01-15",
            time="09:05:00",
            confidence_score=0.87,
            status="Present",
            marked_by="face_recognition",
            created_at="2024-01-15T09:05:00Z",
        )
        assert r.id == 1
        assert r.marked_by == "face_recognition"

    def test_attendance_create_defaults(self):
        c = AttendanceCreate(roll_number="101", name="Alice", date="2024-01-15", time="09:00:00")
        assert c.confidence_score == 0.0
        assert c.status == "Present"
        assert c.marked_by == "face_recognition"

    def test_manual_attendance_defaults(self):
        m = ManualAttendance(roll_number="101", date="2024-01-15", time="08:00:00")
        assert m.status == "Present"

    def test_manual_attendance_roll_number_empty_raises(self):
        with pytest.raises(ValidationError):
            ManualAttendance(roll_number="", date="2024-01-15", time="08:00:00")


# ---------------------------------------------------------------------------
# Recognition models
# ---------------------------------------------------------------------------

class TestRecognitionModels:
    def test_frame_request_valid(self):
        r = FrameRequest(frame="abc123base64")
        assert r.frame == "abc123base64"

    def test_frame_result_valid(self):
        r = FrameResult(
            name="Alice",
            roll_number="101",
            confidence_score=0.9,
            recognition_status="Known",
            bounding_box=(10, 20, 80, 80),
            attendance_marked=True,
            duplicate=False,
        )
        assert r.recognition_status == "Known"
        assert r.bounding_box == (10, 20, 80, 80)

    def test_recognition_response_valid(self):
        result = FrameResult(
            name="Unknown",
            roll_number="",
            confidence_score=0.2,
            recognition_status="Unknown",
            bounding_box=(0, 0, 50, 50),
            attendance_marked=False,
            duplicate=False,
        )
        resp = RecognitionResponse(results=[result], annotated_frame="base64data")
        assert len(resp.results) == 1
        assert resp.annotated_frame == "base64data"

    def test_recognition_response_empty_results(self):
        resp = RecognitionResponse(results=[], annotated_frame="")
        assert resp.results == []


# ---------------------------------------------------------------------------
# Analytics models
# ---------------------------------------------------------------------------

class TestAnalyticsModels:
    def test_dashboard_stats_valid(self):
        d = DashboardStats(
            total_students=45,
            present_today=32,
            absent_today=13,
            attendance_percentage=71.11,
            unknown_face_count=3,
        )
        assert d.total_students == 45

    def test_trend_data_valid(self):
        t = TrendData(date="2024-01-15", count=10)
        assert t.count == 10

    def test_department_stats_valid(self):
        ds = DepartmentStats(department="CS", total=20, present=15, percentage=75.0)
        assert ds.percentage == 75.0
