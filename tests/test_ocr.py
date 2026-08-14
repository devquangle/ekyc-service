import pytest
from ocr import (
    OCRText,
    LayoutParser,
    FieldExtractor,
    CardTypeClassifier,
    MrzParser,
    QrParser,
    parse_date,
    normalize_gender,
    normalize_identity_number,
    normalize_full_name,
    normalize_address,
    normalize_address_for_compare,
    normalize_text_for_compare,
)


def test_spatial_line_grouping():
    parser = LayoutParser()
    tokens = [
        OCRText(text="Họ", confidence=0.98, bbox=[[10, 100], [40, 100], [40, 120], [10, 120]]),
        OCRText(text="và", confidence=0.98, bbox=[[50, 100], [80, 100], [80, 120], [50, 120]]),
        OCRText(text="tên:", confidence=0.98, bbox=[[90, 102], [140, 102], [140, 122], [90, 122]]),
        OCRText(text="TRẦN", confidence=0.97, bbox=[[150, 100], [200, 100], [200, 120], [150, 120]]),
        OCRText(text="THỊ", confidence=0.97, bbox=[[210, 100], [250, 100], [250, 120], [210, 120]]),
        OCRText(text="ÚT", confidence=0.97, bbox=[[260, 100], [290, 100], [290, 120], [260, 120]]),
    ]
    lines = parser.group_tokens_into_lines(tokens)
    assert len(lines) == 1
    assert lines[0].text == "Họ và tên: TRẦN THỊ ÚT"


def test_field_extractor_full_name():
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Họ và tên / Full name:", confidence=0.98, bbox=[[100, 100], [300, 100], [300, 120], [100, 120]]),
        OCRText(text="TRẦN THỊ ÚT", confidence=0.97, bbox=[[310, 100], [450, 100], [450, 125], [310, 125]]),
    ]
    fields = extractor.extract_all_fields(tokens)
    assert "fullName" in fields
    assert fields["fullName"].value == "TRAN THI UT"
    assert fields["fullName"].rawText == "TRẦN THỊ ÚT"
    assert fields["fullName"].bbox == [[310.0, 100.0], [450.0, 100.0], [450.0, 125.0], [310.0, 125.0]]


def test_merged_bbox_calculation():
    extractor = FieldExtractor()
    t1 = OCRText(text="Trần", confidence=0.9, bbox=[[10, 20], [50, 20], [50, 40], [10, 40]])
    t2 = OCRText(text="Văn", confidence=0.9, bbox=[[60, 15], [100, 15], [100, 45], [60, 45]])
    merged = extractor._compute_merged_bbox([t1, t2])
    assert merged == [[10.0, 15.0], [100.0, 15.0], [100.0, 45.0], [10.0, 45.0]]


def test_gender_normalization():
    assert normalize_gender("NAM") == "Nam"
    assert normalize_gender("NỮ") == "Nữ"
    assert normalize_gender("MALE") == "Nam"
    assert normalize_gender("FEMALE") == "Nữ"
    assert normalize_gender("M") == "Nam"
    assert normalize_gender("F") == "Nữ"
    assert normalize_gender(None) is None
    assert normalize_gender("") is None
    assert normalize_gender("   ") is None
    assert normalize_gender("UNKNOWN") is None


def test_date_parser():
    assert parse_date("01/01/1973") == "1973-01-01"
    assert parse_date("01-01-1973") == "1973-01-01"
    assert parse_date("01.01.1973") == "1973-01-01"
    assert parse_date("1973-01-01") == "1973-01-01"
    assert parse_date("730101") == "1973-01-01"

    # Invalid calendar dates MUST return None
    assert parse_date("31/02/1973") is None
    assert parse_date("99/99/1973") is None
    assert parse_date("1973-99-99") is None
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("abc") is None


def test_normalize_identity_number():
    assert normalize_identity_number("086173011002") == "086173011002"
    assert normalize_identity_number("O86173O11OO2") == "086173011002"
    assert normalize_identity_number(" 086 173 011 002 ") == "086173011002"
    assert normalize_identity_number("123456789") == "123456789"
    assert normalize_identity_number("12345") is None
    assert normalize_identity_number("1234567890123") is None
    assert normalize_identity_number(None) is None
    assert normalize_identity_number("") is None


def test_normalize_full_name():
    canonical, raw_clean = normalize_full_name("TRẦN  THỊ   ÚT")
    assert canonical == "TRAN THI UT"
    assert raw_clean == "TRẦN THỊ ÚT"

    c_none, r_none = normalize_full_name(None)
    assert c_none is None and r_none is None


def test_normalize_address():
    norm, raw = normalize_address("123 Đường  Nguyễn Trãi,\n P.5, Q.1")
    assert norm == "123 Đường Nguyễn Trãi, P.5, Q.1"
    assert raw == "123 Đường Nguyễn Trãi, P.5, Q.1"

    cmp = normalize_address_for_compare("123 Đường Nguyễn Trãi, P.5, Q.1")
    assert "123 DUONG NGUYEN TRAI P 5 Q 1" in cmp or "123 DUONG NGUYEN TRAI P5 Q1" in cmp


def test_normalize_text_for_compare():
    assert normalize_text_for_compare(" TRẦN   THỊ ÚT! ") == "TRAN THI UT"
    assert normalize_text_for_compare(None) is None


def test_address_boundary_isolation():
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Quê quán / Place of origin:", confidence=0.98, bbox=[[100, 100], [300, 100], [300, 120], [100, 120]]),
        OCRText(text="Tân Lược, Bình Tân, Vĩnh Long", confidence=0.97, bbox=[[100, 130], [400, 130], [400, 150], [100, 150]]),
        OCRText(text="Có giá trị đến / Date of expiry:", confidence=0.98, bbox=[[100, 160], [350, 160], [350, 180], [100, 180]]),
        OCRText(text="01/01/2033", confidence=0.97, bbox=[[100, 190], [200, 190], [200, 210], [100, 210]]),
        OCRText(text="Nơi thường trú / Place of residence:", confidence=0.98, bbox=[[100, 220], [400, 220], [400, 240], [100, 240]]),
        OCRText(text="Tân Bình, Châu Thành, Đồng Tháp", confidence=0.97, bbox=[[100, 250], [450, 250], [450, 270], [100, 270]]),
    ]
    fields = extractor.extract_all_fields(tokens)

    assert "placeOfOrigin" in fields
    assert fields["placeOfOrigin"].value == "Tân Lược, Bình Tân, Vĩnh Long"
    assert "dateOfExpiry" in fields
    assert fields["dateOfExpiry"].value == "2033-01-01"
    assert "placeOfResidence" in fields
    assert fields["placeOfResidence"].value == "Tân Bình, Châu Thành, Đồng Tháp"


def test_mrz_parser_check_digits():
    parser = MrzParser()
    l1 = "I<VNM0861730110022086173011002<<"
    l2 = "7301017F3301016VNM<<<<<<<<<<<5"
    l3 = "TRAN<<THI<UT<<<<<<<<<<<<<<<<<<"

    result = parser.parse_mrz_lines([l1, l2, l3])
    assert result is not None
    assert result["fullName"] == "TRAN THI UT"
    assert result["dateOfBirth"] == "1973-01-01"
    assert result["dateOfExpiry"] == "2033-01-01"
    assert result["gender"] == "Nữ"


def test_card_type_classifier():
    classifier = CardTypeClassifier()
    front_tokens = [
        OCRText(text="CĂN CƯỚC CÔNG DÂN", confidence=0.98, bbox=[[100, 50], [300, 50], [300, 70], [100, 70]]),
        OCRText(text="Quê quán / Place of origin", confidence=0.98, bbox=[[100, 100], [300, 100], [300, 120], [100, 120]]),
    ]
    card_type, conf = classifier.classify(front_tokens, [], {})
    assert card_type == "CCCD_OLD"
    assert conf >= 0.70


def test_residence_not_lost_when_same_line_as_expiry():
    """
    CCCD_OLD front-side - Huynh Quang Le:
    Mixed layout line contains "Co gia tri den 04/10/2029  Ap Tay".
    placeOfResidence must preserve "Ap Tay" and the subsequent line.
    The stop-label and its date must NOT appear in the extracted value.
    """
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Noi thuong tru / Place of residence:",
                confidence=0.98, bbox=[[10, 270], [380, 270], [380, 290], [10, 290]]),
        OCRText(text="Co gia tri den", confidence=0.96,
                bbox=[[10, 300], [180, 300], [180, 320], [10, 320]]),
        OCRText(text="04/10/2029", confidence=0.97,
                bbox=[[190, 300], [300, 300], [300, 320], [190, 320]]),
        OCRText(text="Ap Tay", confidence=0.97,
                bbox=[[400, 300], [480, 300], [480, 320], [400, 320]]),
        OCRText(text="Tan Binh, Chau Thanh, Dong Thap",
                confidence=0.97, bbox=[[10, 330], [400, 330], [400, 350], [10, 350]]),
    ]
    fields = extractor.extract_all_fields(tokens)
    assert "placeOfResidence" in fields, "placeOfResidence must be extracted"
    val = fields["placeOfResidence"].value
    assert "Tay" in val, f"Address fragment (Ap Tay) missing from value: {val}"
    assert "Binh" in val or "Chau Thanh" in val, f"Address second line missing from value: {val}"
    assert "2029" not in val, f"Date leaked into address value: {val}"


def test_date_of_issue_backside_inline():
    """
    CCCD_NEW back-side - Nguyen Thanh Tung:
    Inline label line: "Ngay, thang, nam cap / Date of issue 23/02/2026"
    Expected: dateOfIssue == "2026-02-23" (ISO)
    """
    extractor = FieldExtractor()
    tokens = [
        OCRText(
            text="Ngay, thang, nam cap / Date of issue 23/02/2026",
            confidence=0.97,
            bbox=[[10, 100], [500, 100], [500, 120], [10, 120]],
        ),
        OCRText(
            text="Ngay, thang, nam het han / Date of expiry 24/12/2028",
            confidence=0.97,
            bbox=[[10, 130], [500, 130], [500, 150], [10, 150]],
        ),
    ]
    fields = extractor.extract_all_fields(tokens)
    assert "dateOfIssue" in fields, "dateOfIssue must be extracted from back-side inline label"
    assert fields["dateOfIssue"].value == "2026-02-23", \
        f"Expected 2026-02-23, got: {fields['dateOfIssue'].value}"
    assert "dateOfExpiry" in fields
    assert fields["dateOfExpiry"].value == "2028-12-24", \
        f"Expected 2028-12-24, got: {fields['dateOfExpiry'].value}"


def test_cccd_new_back_address_fields():
    """
    CCCD_NEW back-side - Nguyen Thanh Tung:
    placeOfResidence spans 2 lines, placeOfOrigin (birth registration) is on its own line.
    """
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Noi cu tru / Place of residence:",
                confidence=0.98, bbox=[[10, 50], [350, 50], [350, 70], [10, 70]]),
        OCRText(text="To 09, Ap Phu Binh",
                confidence=0.97, bbox=[[10, 80], [300, 80], [300, 100], [10, 100]]),
        OCRText(text="Tan Phu Trung, Dong Thap",
                confidence=0.97, bbox=[[10, 110], [300, 110], [300, 130], [10, 130]]),
        OCRText(text="Noi dang ky khai sinh / Place of birth registration:",
                confidence=0.98, bbox=[[10, 140], [450, 140], [450, 160], [10, 160]]),
        OCRText(text="Tan Phu Trung, Dong Thap",
                confidence=0.97, bbox=[[10, 170], [300, 170], [300, 190], [10, 190]]),
    ]
    fields = extractor.extract_all_fields(tokens)
    assert "placeOfResidence" in fields
    res_val = fields["placeOfResidence"].value
    assert "Phu" in res_val, f"placeOfResidence incomplete: {res_val}"
    assert "placeOfOrigin" in fields
    orig_val = fields["placeOfOrigin"].value
    assert "Tan Phu Trung" in orig_val, f"placeOfOrigin incorrect: {orig_val}"


def test_separate_label_and_value_boxes_cccd_old():
    """
    Test separate label_box and value_box extraction for CCCD_OLD front layout:
    - identityNumber: inline (label left, value right)
    - fullName: stacked (label row 1, value row 2)
    - gender & nationality on single row: 4 distinct boxes
    - placeOfResidence: 2 lines
    """
    extractor = FieldExtractor()
    tokens = [
        # identityNumber inline
        OCRText(text="Số / No.:", confidence=0.98, bbox=[[300, 100], [420, 100], [420, 130], [300, 130]]),
        OCRText(text="087204000897", confidence=0.99, bbox=[[430, 100], [800, 100], [800, 130], [430, 130]]),

        # fullName stacked
        OCRText(text="Họ và tên / Full name:", confidence=0.98, bbox=[[300, 150], [550, 150], [550, 175], [300, 175]]),
        OCRText(text="HUỲNH QUANG LÊ", confidence=0.99, bbox=[[300, 185], [600, 185], [600, 215], [300, 215]]),

        # dateOfBirth inline
        OCRText(text="Ngày sinh / Date of birth:", confidence=0.98, bbox=[[300, 230], [560, 230], [560, 255], [300, 255]]),
        OCRText(text="04/10/2004", confidence=0.99, bbox=[[570, 230], [700, 230], [700, 255], [570, 255]]),

        # gender & nationality on same row: [Label Giới tính] [Value Giới tính] [Label Quốc tịch] [Value Quốc tịch]
        OCRText(text="Giới tính / Sex:", confidence=0.98, bbox=[[300, 270], [430, 270], [430, 295], [300, 295]]),
        OCRText(text="Nam", confidence=0.99, bbox=[[440, 270], [490, 270], [490, 295], [440, 295]]),
        OCRText(text="Quốc tịch / Nationality:", confidence=0.98, bbox=[[550, 270], [750, 270], [750, 295], [550, 295]]),
        OCRText(text="Việt Nam", confidence=0.99, bbox=[[760, 270], [850, 270], [850, 295], [760, 295]]),

        # placeOfOrigin
        OCRText(text="Quê quán / Place of origin:", confidence=0.98, bbox=[[300, 310], [560, 310], [560, 335], [300, 335]]),
        OCRText(text="Tân Bình, Châu Thành, Đồng Tháp", confidence=0.97, bbox=[[300, 345], [780, 345], [780, 370], [300, 370]]),

        # placeOfResidence line 1
        OCRText(text="Nơi thường trú / Place of residence:", confidence=0.98, bbox=[[300, 385], [620, 385], [620, 410], [300, 410]]),
        OCRText(text="Ấp Tây", confidence=0.97, bbox=[[630, 385], [720, 385], [720, 410], [630, 410]]),

        # shared bottom row: Expiry (left) and Address line 2 (right)
        OCRText(text="Có giá trị đến / Date of expiry: 04/10/2029", confidence=0.98, bbox=[[20, 430], [320, 430], [320, 455], [20, 455]]),
        OCRText(text="Tân Bình, Châu Thành, Đồng Tháp", confidence=0.97, bbox=[[400, 430], [850, 430], [850, 455], [400, 455]]),
    ]

    fields = extractor.extract_all_fields(tokens)

    # 1. Identity Number
    assert "identityNumber" in fields
    id_f = fields["identityNumber"]
    assert id_f.label_box is not None and id_f.value_box is not None
    assert id_f.label_box[2] <= id_f.value_box[0] + 20, "Identity number label box must be on the left of value box"

    # 2. Full Name
    assert "fullName" in fields
    name_f = fields["fullName"]
    assert name_f.label_box is not None and name_f.value_box is not None
    assert name_f.label_box[3] <= name_f.value_box[1] + 15, "Full name label box must be above value box"

    # 3. Gender & Nationality 4 distinct boxes
    assert "gender" in fields and "nationality" in fields
    g_f = fields["gender"]
    nat_f = fields["nationality"]
    assert g_f.label_box is not None and g_f.value_box is not None
    assert nat_f.label_box is not None and nat_f.value_box is not None
    assert g_f.label_box[0] < g_f.value_box[0]
    assert g_f.value_box[0] < nat_f.label_box[0]
    assert nat_f.label_box[0] < nat_f.value_box[0]

    # 4. Residence & Expiry
    assert "placeOfResidence" in fields
    res_f = fields["placeOfResidence"]
    assert res_f.label_box is not None and res_f.value_box is not None
    assert ("Tây" in res_f.value or "Tay" in res_f.value) and ("Bình" in res_f.value or "Binh" in res_f.value)

    assert "dateOfExpiry" in fields
    exp_f = fields["dateOfExpiry"]
    assert exp_f.value == "2029-10-04"
    assert exp_f.value_box is not None


def test_separate_label_and_value_boxes_cccd_new():
    """
    Test CCCD_NEW stacked layout boxes.
    """
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Số định danh cá nhân / Personal identification number", confidence=0.98, bbox=[[300, 100], [800, 100], [800, 125], [300, 125]]),
        OCRText(text="087203001336", confidence=0.99, bbox=[[300, 135], [600, 135], [600, 165], [300, 165]]),

        OCRText(text="Họ, chữ đệm và tên khai sinh / Full name:", confidence=0.98, bbox=[[300, 180], [750, 180], [750, 205], [300, 205]]),
        OCRText(text="NGUYỄN THANH TÙNG", confidence=0.99, bbox=[[300, 215], [650, 215], [650, 245], [300, 245]]),

        OCRText(text="Ngày, tháng, năm sinh / Date of birth:", confidence=0.98, bbox=[[300, 260], [680, 260], [680, 285], [300, 285]]),
        OCRText(text="24/12/2003", confidence=0.99, bbox=[[300, 295], [450, 295], [450, 320], [300, 320]]),
    ]
    fields = extractor.extract_all_fields(tokens)
    assert fields["identityNumber"].label_box[3] <= fields["identityNumber"].value_box[1] + 15
    assert fields["fullName"].label_box[3] <= fields["fullName"].value_box[1] + 15
    assert fields["dateOfBirth"].label_box[3] <= fields["dateOfBirth"].value_box[1] + 15
