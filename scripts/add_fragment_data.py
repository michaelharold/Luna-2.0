import json

# Terse, fragment-style phrasings (bare noun phrases, not full questions) for facts
# already in the dataset. All existing training data is phrased as full questions
# ("Who is the Senior Advisor...?"), so a bare fragment like "Senior advisor." doesn't
# match well and falls back to refusal even though the fact exists. These pairs teach
# that same short-utterance style directly.
FRAGMENT_PAIRS = [
    ("Name of the college.", "TKM College of Engineering, Kollam."),
    ("Full name.", "Thangal Kunju Musaliar College of Engineering."),
    ("Location.", "TKM College of Engineering is located at Karikode, Kollam, Kerala, India."),
    ("Nearby landmarks.", "The college is located at a distance of 450 meters from Kilikollur railway station."),
    ("PIN code.", "691005."),
    ("Established.", "TKM College of Engineering was established in 1958."),
    ("Founder.", "TKM College of Engineering was founded by Janab A. Thangal Kunju Musaliar."),
    ("Autonomous status.", "Yes, TKM College of Engineering has been autonomous since May 2022."),
    ("NAAC grade.", "The college has been NAAC accredited with an A grade."),
    ("NBA accreditation.", "Several undergraduate engineering (B.Tech) programs are accredited, including Mechanical, Electrical & Electronics, Electronics & Communication, and Computer Science and Engineering."),
    ("KIRF ranking.", "The college is ranked second in Kerala among engineering colleges by the Kerala Institutional Ranking Framework."),
    ("Vision.", "The vision of the department is to be a centre of excellence imparting quality education in Computer Science and Engineering and transforming students into critical thinkers and lifelong learners capable of developing environment-friendly and economically feasible solutions to real-world problems."),
    ("Mission.", "The mission of the department is to provide a strong foundation in Computer Science and Engineering, prepare students for professional careers and higher education, inculcate research interest, stay abreast of technological advances, and impart skills to create socially acceptable solutions upholding ethical values."),
    ("HOD.", "Dr. Aneesh G Nath is the Head of the Computer Science Department."),
    ("Head of department.", "Dr. Aneesh G Nath is the Head of the Computer Science Department."),
    ("Senior advisor.", "Dr. Dimple A Shajahan is the Senior Advisor of the Computer Science Department."),
    ("AI department advisor.", "Dr. Thushara is the Advisor of the Computer Science with AI Department."),
    ("AI Lab.", "The AI Lab is located on the second floor of the Computer Science Department."),
    ("Operating systems lab.", "The Operating Systems Lab is located on the ground floor of the Computer Science Department."),
    ("Systems software lab.", "The Systems Software Lab is located on the second floor of the Computer Science Department."),
    ("Computing lab.", "The Computing Lab is located on the second floor of the Computer Science Department."),
    ("HOD's room.", "The HOD's room is located on the ground floor."),
    ("Faculty room.", "The faculty room is located on the second floor."),
    ("KTU office.", "The KTU office is on the first floor of the Computer Science Department."),
    ("Seminar hall.", "The seminar hall is on the ground floor of the Computer Science Department."),
    ("Jubilee hall.", "Jubilee hall is on the first floor."),
    ("Ladies' common room.", "The ladies' common room is at the backside of the college canteen."),
    ("Principal's room.", "The principal's room is on the first floor of the administrative block."),
    ("College canteen.", "The college canteen and store are opposite the Somanathan Centre near the back gate."),
    ("Campus doctor.", "The campus doctor is located near the college store, near the back gate."),
    ("Printout.", "Printout facilities are available at the college store."),
    ("Nearest railway station.", "Kollam railway station is at a distance of 7 kilometers, and Kilikollur railway station is at a distance of 500 meters from the college."),
    ("Technical clubs.", "The best technical clubs and societies include Coding Club, CSI, IEEE, ISTE, IEI, and Civil Service Aspirants Club."),
    ("Extracurricular clubs.", "Extracurricular clubs include Dance Club, Music Club, TKM Photography Club, Film and Drama Club, Literary and Debate Club, Fine Arts Club, and College Sports Council."),
    ("Hestia.", "Hestia is the annual techno-cultural festival of TKM College of Engineering, bringing together students to participate in technical and cultural events."),
    ("Placement rate.", "TKM College of Engineering has a placement rate of approximately 90%."),
    ("Companies for placement.", "More than 100 companies visit TKM College of Engineering for campus placements every year."),
    ("Highest package.", "The highest package offered is ₹21 LPA."),
    ("Average package.", "The average package offered is ₹5.03 LPA."),
    ("Placement cell.", "Yes. TKM College of Engineering has a Career Guidance and Placement Unit (CGPU)."),
    ("CGPU.", "The Career Guidance and Placement Unit (CGPU) is responsible for career guidance, placement training, and campus recruitment activities."),
    ("Placement training.", "Yes, TKM College of Engineering provides placement training to prepare students for campus recruitment."),
    ("Campus placements.", "Yes, TKM College of Engineering conducts campus placement drives with leading companies."),
    ("Placement facilities.", "The placement unit has air-conditioned halls, computer laboratories, and interview rooms for recruitment activities."),
    ("Ragging.", "Yes, the campus is strictly ragging free."),
]

with open("data/train_data.jsonl", "r", encoding="utf-8") as f:
    existing = [json.loads(line) for line in f if line.strip()]

existing_keys = {(e["instruction"].strip().lower(), e["response"].strip().lower()) for e in existing}

added = 0
with open("data/train_data.jsonl", "a", encoding="utf-8") as f:
    for instruction, response in FRAGMENT_PAIRS:
        key = (instruction.strip().lower(), response.strip().lower())
        if key in existing_keys:
            continue
        f.write(json.dumps({"instruction": instruction, "response": response}, ensure_ascii=False) + "\n")
        existing_keys.add(key)
        added += 1

print(f"Added {added} fragment-style pairs. Total dataset size: {len(existing) + added}")
