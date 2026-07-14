'''
take the student details 
'''
from dataclasses import dataclass
from typing import Optional

@dataclass
class StudentDetails:
    #details of the student 
    name: str
    roll_number: str
    department: Optional[str] = None
    department: Optional[str] = None 
    
    def folder_name(self) -> str:
        #build a system to store photo
        safe_name = "".join(c for c in self.name if c.isalnum() or c in ("_", "-")).strip()
        safe_roll = "".join(c for c in self.roll_number if c.isalnum() or c in ("_","-")).strip()
        return f"{safe_name}_{safe_roll}" if safe_name else safe_roll
    
    
@dataclass
class CaptureResult:
    #Summary returned after a capture session 
    student: StudentDetails
    images_saved: int
    target_images: int
    folder_path: str
    completed: bool
    message: str = ""