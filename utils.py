import re

def load_frame_to_altitude_map(srt_path: str) -> dict:
    with open(srt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    frame_data = []
    current_frame = None

    for line in lines:
        line = line.strip()

        if re.match(r"^\d{4,5}$", line):
            current_frame = int(line)

        elif "rel_alt" in line:
            alt_match = re.search(r"rel_alt: ([\d.]+)", line)
            if alt_match and current_frame is not None:
                rel_alt = float(alt_match.group(1))
                frame_data.append((current_frame, rel_alt))

    return {frame: alt for frame, alt in frame_data}
