# Scene DSL Grammar

## Overview

This document describes the grammar for the scene description domain-specific language (DSL). The DSL is designed to be deterministic, with a fixed vocabulary and no ambiguity.

## Vocabulary

### Special Tokens
- `<BOS>` (id: 0) - Beginning of sequence
- `<EOS>` (id: 1) - End of sequence
- `<PAD>` (id: 2) - Padding token

### Content Tokens
- **Colors**: red, green, blue, yellow
- **Shapes**: ball, cube, block
- **Numbers**: one, two, three, four, five
- **Relations**: left of, right of, on, in front of
- **Function words**: There, is, are, the, a, of, on, in, front, .
- **Keywords**: table (reserved for future use)

Total vocabulary size: 28 tokens

## Grammar

### Top-Level Structure

```ebnf
S := MAIN "."
MAIN := COUNT_SENT | REL_SENT
```

Every valid sentence is either a COUNT sentence or a RELATIONAL sentence, and must end with a period.

### COUNT Sentences

Describe a quantity of objects with a single color and shape.

```ebnf
COUNT_SENT := "There" VERB_COUNT NUMBER COLOR SHAPE PLURAL_OPT
VERB_COUNT := "is" | "are"
PLURAL_OPT := "" | "s"
```

**Semantic constraints:**
- Singular (count=1): use "is", no plural 's'
- Plural (count≥2): use "are", add plural 's'

**Examples:**
```
There is one red ball.
There are two green cubes.
There are five yellow blocks.
```

### RELATIONAL Sentences

Describe a spatial relationship between two objects.

```ebnf
REL_SENT := "The" COLOR SHAPE VERB_REL REL "the" COLOR SHAPE
VERB_REL := "is" | "are"
REL := "left of" | "right of" | "on" | "in front of"
```

**Semantic constraints (v1):**
- Both objects have count=1
- Use "is" for singular objects

**Examples:**
```
The red ball is left of the blue cube.
The green block is on the yellow ball.
The blue cube is in front of the red block.
```

### Terminal Symbols

```ebnf
NUMBER := "one" | "two" | "three" | "four" | "five"
COLOR := "red" | "green" | "blue" | "yellow"
SHAPE := "ball" | "cube" | "block"
```

## Scene Graph Mapping

Each valid sentence maps deterministically to a scene graph JSON structure:

```json
{
  "canvas": {"W": 64, "H": 64, "bg": "white"},
  "objects": [
    {"id": "o1", "shape": "ball", "color": "red", "count": 1}
  ],
  "relations": [],
  "constraints": {"no_overlap": true, "min_dist_px": 4, "layout": "pack"}
}
```

### COUNT_SENT Mapping
- Single object in `objects` list
- `count` field reflects the number word (1-5)
- Empty `relations` list

### REL_SENT Mapping
- Two objects in `objects` list (both with count=1)
- One relation in `relations` list
- Relation types: `left_of`, `right_of`, `on`, `in_front_of`

## Canonical Form

The canonical form ensures deterministic text ↔ scene graph conversion:

1. **Text → Scene Graph**: Use parser to deterministically extract objects and relations
2. **Scene Graph → Text**: Use canonicalizer to generate text in standard order
3. **Round-trip property**: `parse(canonicalize(parse(text))) = parse(text)`

## Design Principles

1. **Deterministic**: No ambiguity in parsing or generation
2. **Finite**: Closed vocabulary, enumerable sample space
3. **Compositional**: Separate dimensions (color, shape, count, relation) for systematic generalization testing
4. **Minimal**: Small vocabulary (28 tokens) for efficient training from scratch
5. **Grammatical**: Enforces agreement (singular/plural) and valid structure

## Future Extensions (v2+)

Potential extensions not included in v1:
- Multiple relations per scene
- Plural objects in relations
- Additional shapes/colors
- Spatial modifiers (e.g., "far", "near")
- Scene backgrounds beyond "table"
