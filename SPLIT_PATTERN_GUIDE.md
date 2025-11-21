# Split Pattern Feature Guide

## Overview

The Split Pattern feature allows the AI to automatically parse single "x-y" values written on notes and split them into two separate columns in your CSV output.

## Problem Solved

**Before:** You write "3-5" on your note, and you want:
- Column E (Verdivurdering) = 3
- Column F (Gjennomførbarhet) = 5

**Challenge:** GenAI vision models don't inherently understand this parsing rule.

**Solution:** The tool automatically detects this pattern and instructs Claude to split the value correctly.

## How It Works

### 1. Automatic Detection

When you load a template with these columns:
- **Verdivurdering** (or similar: verdi, value, priority)
- **Gjennomførbarhet** (or similar: feasibility, complexity)

The tool automatically detects if they're adjacent and enables split pattern mode.

### 2. AI Instruction

The tool generates a detailed prompt for Claude:

```
⚠️ CRITICAL SPLIT PATTERN RULE:
When you see a value like "3-5" or "x-y" on the note:
  - Extract the FIRST number (before dash) → "Verdivurdering"
  - Extract the SECOND number (after dash) → "Gjennomførbarhet"
  - Do NOT put "3-5" in both fields - split them!
```

### 3. Example Extraction

**Physical Note:**
```
Bedrift: Acme Corp
Implementere nytt CRM-system
Business Value: Øke kundelojalitet
Risk: Integrasjonskompleksitet
3-5
```

**AI Extracts:**
```json
{
  "Bedrift": "Acme Corp",
  "Beskrivelse": "Implementere nytt CRM-system",
  "Business Value": "Øke kundelojalitet",
  "Risk": "Integrasjonskompleksitet",
  "Verdivurdering": "3",
  "Gjennomførbarhet": "5"
}
```

**CSV Output:**
```csv
Bedrift,Beskrivelse,Business Value,Risk,Verdivurdering,Gjennomførbarhet
Acme Corp,Implementere nytt CRM-system,Øke kundelojalitet,Integrasjonskompleksitet,3,5
```

## Configuration

### Automatic (Recommended)

When you load a template:
```bash
📁 Template file path: example_template.csv

✓ Found 6 columns in template

Configure fields manually? (y/N): [press Enter]

✓ Using smart defaults for field configuration

📊 Split Pattern Detected:
  On the note: A single value like '3-5' will be split:
    • Verdivurdering = 3 (first number)
    • Gjennomførbarhet = 5 (second number)
```

### Manual Configuration

If you choose manual configuration:
```bash
Configure fields manually? (y/N): y

[... configure each field ...]

Split Pattern Configuration
Do any two adjacent fields share a single 'x-y' value on the note?
(e.g., '3-5' where 3→Field1, 5→Field2)
Configure split pattern? (y/N): y

Available fields:
  [1] Bedrift
  [2] Beskrivelse
  [3] Business Value
  [4] Risk
  [5] Verdivurdering
  [6] Gjennomførbarhet

First field (number before dash): 5
Second field (number after dash): 6

✓ Split pattern: Verdivurdering (x) ← 'x-y' → Gjennomførbarhet (y)
```

## Supported Formats

The AI is trained to recognize **multiple real-world notation styles** that people actually use:

### Common Formats (All Supported)

| Format | Example | Extracted As | Notes |
|--------|---------|--------------|-------|
| **Dash** | `3-5` | V=3, F=5 | Most common, recommended |
| **Slash** | `3/5` | V=3, F=5 | Alternative separator |
| **Space** | `V3 F5` | V=3, F=5 | With field indicators |
| **Lowercase** | `v3 f5` | V=3, F=5 | Case insensitive |
| **Colon** | `v:3 f:5` | V=3, F=5 | Field:value format |
| **Uppercase** | `V:3 F:5` | V=3, F=5 | Field:value format |
| **Prefix dash** | `v2-v3` | V=2, F=3 | With repeated prefix |

### How It Works

The AI prompt includes examples of all these formats, so Claude understands:
- Look for **two numbers** somewhere on the note
- They might have letters (v, f, V, F) as indicators
- They might be separated by dash, slash, space, or colon
- Extract just the numbers and put them in separate fields

### Real-World Examples

**Note 1:**
```
Bedrift: Acme
New CRM
Risk: Low
v3-f5          ← AI extracts: V=3, F=5
```

**Note 2:**
```
Company: Beta
API Integration
V:4 F:2        ← AI extracts: V=4, F=2
```

**Note 3:**
```
Customer: Gamma
Cloud migration
2/3            ← AI extracts: V=2, F=3
```

All work! The AI is flexible enough to handle human handwriting variations.

### What If Format Isn't Recognized?

The validation system will catch it:
```
⚠ sticky_005.jpg note #2: 'Verdivurdering' contains 'v3-f5'
   (looks like unsplit pattern - should be two separate numbers)
```

Image goes to `review/` folder for manual inspection.

## Field Requirements

For split pattern to activate automatically:
1. **Field Names:** Must contain keywords like:
   - First field: "verdivurdering", "verdi", "value", "priority"
   - Second field: "gjennomførbarhet", "feasibility", "complexity"

2. **Position:** Fields should be adjacent in the template

3. **Order:** Typically Verdivurdering (E) before Gjennomførbarhet (F)

## Multi-Note Support

Split patterns work seamlessly with multi-note detection:

**Image with 3 sticky notes, each with "x-y" values:**

Note 1: `2-4` → Verdivurdering=2, Gjennomførbarhet=4
Note 2: `3-5` → Verdivurdering=3, Gjennomførbarhet=5
Note 3: `1-3` → Verdivurdering=1, Gjennomførbarhet=3

**CSV Output:**
```csv
Bedrift,Beskrivelse,Business Value,Risk,Verdivurdering,Gjennomførbarhet
Company A,Project A,High value,Low risk,2,4
Company B,Project B,Medium value,Medium risk,3,5
Company C,Project C,Low value,High risk,1,3
```

## Testing Recommendations

1. **Start with Preview Mode:**
   ```bash
   uv run python main.py
   # Select: [2] Preview first 5 images
   ```

2. **Verify Split Pattern Detection:**
   - Check console output for "📊 Split Pattern Detected"
   - Review Field Configuration table

3. **Inspect First Results:**
   - Open `output.csv` after preview
   - Verify numbers are split correctly
   - Check that no "x-y" values appear in output

4. **Adjust if Needed:**
   - If split not detected: Use manual configuration
   - If extraction incorrect: Check note clarity/formatting

## Troubleshooting

### Pattern Not Detected

**Problem:** Auto-detection didn't find split pattern

**Solutions:**
1. Use manual configuration mode
2. Ensure field names contain keywords (verdivurdering, gjennomførbarhet)
3. Check fields are adjacent in template

### Incorrect Splitting

**Problem:** AI puts "3-5" in both columns or doesn't split

**Solutions:**
1. Ensure note has clear "x-y" format with dash
2. Try adding space around dash: "3 - 5"
3. Test with Preview mode first
4. Check AI prompt in logs to verify instructions

### Mixed Results

**Problem:** Some notes split correctly, others don't

**Solutions:**
1. Ensure consistent notation across all notes
2. Check note clarity (good lighting, clear handwriting)
3. Verify no other text near the "x-y" value
4. Consider using printed notes for testing

## Advanced: Custom Split Patterns

You can adapt this for other split patterns by modifying field detection:

**Example:** Split "Customer/Project" into two fields:

1. Manual configuration mode
2. Configure split pattern for those fields
3. Modify prompt template if needed

## Cost Efficiency

**No additional cost:** Split pattern uses the same single API call as regular extraction. The intelligence is in the prompt, not in extra processing.

## Next Steps

1. ✅ Create your CSV template with Verdivurdering and Gjennomførbarhet columns
2. ✅ Write notes using "x-y" format (e.g., "3-5")
3. ✅ Run tool with auto-configuration
4. ✅ Verify split pattern detected
5. ✅ Process images and verify CSV output

For more information, see README.md.
