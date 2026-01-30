# Pace / Lap Source Notes

## JV-Data structure sources
- C# struct (SDK v4.9.0.2):
  - `C:\Users\yyosh\Downloads\JVDTLABSDK4902\JRA-VAN Data Lab. SDK Ver4.9.0.2\JV-Data構造体\C#版\JVData_Struct.cs`
- VB struct (SDK v4.9.0.2):
  - `C:\Users\yyosh\Downloads\JVDTLABSDK4902\JRA-VAN Data Lab. SDK Ver4.9.0.2\JV-Data構造体\VB2019版\JVData_Structure.vb`

Note: Not found inside the repo or under `C:\Program Files` / `C:\Program Files (x86)` during search.

## RA record (JV_RA_RACE) lap/pace fields
Source: `JV_RA_RACE` in `JVData_Struct.cs`.

- Record length: 1272 bytes.
- LapTime array:
  - `LapTime[25]`
  - 1-based offset: 891 + (3 * i), length 3 bytes each (i=0..24)
- Haron (3F/4F) times (1-based offsets, length 3 bytes each):
  - HaronTimeS3: 970
  - HaronTimeS4: 973
  - HaronTimeL3: 976
  - HaronTimeL4: 979

## Units and missing codes
- The struct file does not explicitly state missing codes for LapTime fields.
- The current parser (`py32_fetcher/parsers/race_parser.py`) treats:
  - `<= 0` as missing
  - `>= 999` as missing
  - Otherwise, value is `value / 10.0` seconds (0.1-second units).

## Distance remainder handling
- The struct does not document how LapTime behaves when distance is not a multiple of 200m.
- For now, LapTime is used as provided without normalization.
