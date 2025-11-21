# Quick Reference Card

## ⚡ Quick Start

```bash
# 1. Setup
uv sync
echo "ANTHROPIC_API_KEY=your-key" > .env

# 2. Add images to input/
# 3. Run
uv run python main.py

# 4. Choose Preview mode (test first 5 images)
# 5. Check output.csv and review/ folder
```

---

## 📁 Folder System

| Folder | Status | Meaning |
|--------|--------|---------|
| **input/** | 📥 Source | Your images to process |
| **processed/** | ✅ Success | Clean extraction, no issues |
| **review/** | ⚠️ Warning | Data saved but quality suspicious |
| **failed/** | ❌ Error | Complete failure, no usable data |

---

## ⚙️ Configuration Quick Tweaks

**Location:** `config.py`

### Common Adjustments

```python
# Be more lenient with quality
MAX_NULL_FIELDS_RATIO = 0.7      # Allow 70% null fields (vs 50% default)
WARN_ON_MOSTLY_NULL = False      # Don't flag null fields

# Expect specific number of notes
MIN_EXPECTED_NOTES = 3           # Expect at least 3 notes per image

# Strict mode (reject partial failures)
ACCEPT_PARTIAL_EXTRACTION = False   # Don't save partial data
REQUIRE_ALL_FIELDS = True           # All fields must be filled
```

---

## 🚨 What Goes to Review vs Failed

### Review Folder ⚠️
*Data IS saved to CSV, but needs manual check*

- Empty extraction (`[]` returned)
- Fewer notes than expected
- >50% fields are null
- Split pattern incomplete ("3-5" not split)
- Missing non-required fields

### Failed Folder ❌
*NO data saved, complete failure*

- API errors after 3 retries
- Invalid JSON response
- Corrupt/unreadable image
- Network timeout
- Authentication failure

---

## 📊 Summary Output Explained

```
Images:
  ✓ Processed: 18          ← Moved to processed/
  ⚠ Review needed: 4       ← Moved to review/ - CHECK THESE!
  ✗ Failed: 3             ← Moved to failed/ - retry or delete
  Success rate: 86.4%      ← (18+4)/(18+4+3)

Notes extracted: 62        ← Total rows in output.csv
  Average per image: 3.4   ← 62/18 images

⚠ Warnings (7):
  • Mostly null fields: 3  ← Check review/ for these
  • Incomplete split: 2    ← x-y not parsed correctly
```

---

## 🔍 Common Warning Types

| Warning | Cause | Action |
|---------|-------|--------|
| **Empty extraction** | No notes found | Check if image is blank or illegible |
| **Mostly null fields** | >50% fields empty | Verify note has all info written |
| **Incomplete split** | x-y not split into two columns | Check if "3-5" is clear on note |
| **Split not applied** | Still contains "3-5" in one field | AI didn't parse dash correctly |

---

## 🎯 Split Pattern Reference

**The AI recognizes multiple formats:**

| Your Note Shows | Extracted As |
|-----------------|--------------|
| `3-5` | V=3, F=5 |
| `V3 F5` | V=3, F=5 |
| `v:3 f:5` | V=3, F=5 |
| `2/4` | V=2, F=4 |
| `v2-v3` | V=2, F=3 |

**Expected CSV:**
```csv
Verdivurdering,Gjennomførbarhet
3,5
```

**If you see this (WRONG):**
```csv
Verdivurdering,Gjennomførbarhet
v3-f5,null
```
→ Check `review/` folder - validation caught unsplit pattern

---

## 💡 Best Practices

### Testing New Batches
1. ✅ Always start with **Preview mode** (5 images)
2. ✅ Check `review/` folder manually
3. ✅ Verify `output.csv` looks correct
4. ✅ Adjust `config.py` if needed
5. ✅ Then run **Full Batch**

### During Production
1. ✅ Monitor `review/` folder after each run
2. ✅ Check `warnings.log` for patterns
3. ✅ If >20% in review → improve image quality
4. ✅ If >10% in failed → check API or image format

### Handling Review Folder
1. Inspect each image in `review/`
2. If extraction is correct → move to `processed/`
3. If partially correct → manually fix `output.csv`
4. If wrong → delete from CSV, retake photo

---

## 🔧 Troubleshooting

### Everything goes to review/
**Fix:** Relax thresholds
```python
MAX_NULL_FIELDS_RATIO = 0.8
MIN_EXPECTED_NOTES = 0
```

### Too many API failures
**Check:**
- API key is valid
- Not rate limited
- Network connection
- `logs/process.log` for error details

### Split pattern not working
**Verify:**
1. Columns are named "Verdivurdering" and "Gjennomførbarhet"
2. They are adjacent in template
3. Note shows dash clearly: "3-5" not "3/5" or "3 5"

---

## 📝 Files to Check

| File | Purpose |
|------|---------|
| `output.csv` | All extracted data (main result) |
| `warnings.log` | All validation warnings |
| `logs/process.log` | Complete processing log |
| `review/` | Images needing manual verification |
| `failed/` | Images that failed completely |

---

## 🎛️ Processing Modes

1. **Dry Run** - Estimate cost only (no API calls)
2. **Preview** - Process first 5 images (test before full batch)
3. **Full Batch** - Process all images
4. **Resume** - Continue after interruption

---

## ⏱️ Typical Workflow

```
10:00 - Load template, configure split pattern (2 min)
10:02 - Run Preview mode on 5 images (1 min)
10:03 - Check output.csv and review/ folder (2 min)
10:05 - Adjust config if needed (1 min)
10:06 - Run Full Batch on remaining images (5 min for 50 images)
10:11 - Review warnings summary (1 min)
10:12 - Manually check review/ folder (5 min for 3-4 images)
10:17 - DONE!
```

---

## 💰 Cost Estimate

- ~$0.01-0.02 per image (Claude Sonnet 4)
- 100 images ≈ $1-2
- Multi-note images = same cost (single API call)

---

## 🆘 Getting Help

1. Check `logs/process.log` for detailed errors
2. Review `warnings.log` for extraction issues
3. See `VALIDATION_GUIDE.md` for complete validation docs
4. See `SPLIT_PATTERN_GUIDE.md` for split pattern details
5. Check `README.md` for full documentation
