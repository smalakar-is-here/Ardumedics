"""
ArduMedics Prescription Parser

Extracts medicine names, dosages, and schedules from OCR text.
"""

import re


class PrescriptionParser:
    """
    Parse OCR text from medical prescriptions to extract structured information.

    Extracts:
    - Medicine names (tab, cap, syp, inj, etc.)
    - Dosages (mg, ml, mcg, IU)
    - Schedules (daily, BD, TDS, SOS, PRN, etc.)
    """

    # Medicine form patterns
    FORMS = r'(?i)(tab|cap|syp|syrup|inj|cream|gel|drops|ointment|lotion|powder|sachet)'

    # Dosage patterns
    DOSAGE = r'(?i)(\d+\.?\d*)\s*(mg|ml|mcg|iu|g|units?)'

    # Schedule patterns
    SCHEDULES = {
        'OD': 'once daily',
        'BD': 'twice daily',
        'TDS': 'three times daily',
        'QID': 'four times daily',
        'SOS': 'as needed',
        'PRN': 'as needed',
        'STAT': 'immediately',
        'HS': 'at bedtime',
        'AC': 'before meals',
        'PC': 'after meals',
        'QD': 'every day',
        'QOD': 'every other day',
        'QWK': 'every week',
    }

    def parse(self, ocr_text):
        """
        Parse OCR text into structured prescription data.

        Args:
            ocr_text: Raw text from OCR engine

        Returns:
            dict with: medicines, dosages, schedules, raw_text
        """
        medicines = self._extract_medicines(ocr_text)
        dosages = self._extract_dosages(ocr_text)
        schedules = self._extract_schedules(ocr_text)

        return {
            'medicines': medicines,
            'dosages': dosages,
            'schedules': schedules,
            'raw_text': ocr_text
        }

    def _extract_medicines(self, text):
        """Extract medicine names from text."""
        medicines = []

        # Pattern: [form] [name]
        pattern = f'{self.FORMS}\\.?\\s+([A-Za-z][A-Za-z0-9\\-]+)'
        matches = re.findall(pattern, text)

        for form, name in matches:
            medicines.append({
                'name': name.strip(),
                'form': form.upper()
            })

        return medicines

    def _extract_dosages(self, text):
        """Extract dosages from text."""
        dosages = []

        pattern = self.DOSAGE + r'(?:\s*(\d+[xd]))?'
        matches = re.findall(pattern, text, re.IGNORECASE)

        for value, unit, freq in matches:
            dosages.append({
                'value': float(value),
                'unit': unit.lower(),
                'frequency': freq if freq else None
            })

        return dosages

    def _extract_schedules(self, text):
        """Extract medication schedules from text."""
        schedules = []

        text_upper = text.upper()
        for abbr, full in self.SCHEDULES.items():
            if re.search(r'\b' + abbr + r'\b', text_upper):
                schedules.append({
                    'abbreviation': abbr,
                    'meaning': full
                })

        return schedules
