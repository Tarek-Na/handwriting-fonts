# Notes for Claude

Start with [instruction.md](instruction.md): the project's state, its rules,
what has been measured and ruled out, and what to do next. The earlier
conversations are not available; that file carries what they established.

The rules that matter most, in brief:

- The repository is public. Never commit a photo of real handwriting or any
  image made from one (crops, overlays, letter sheets, model outputs);
  `.gitignore` covers the known paths, and `git status` must show none before
  a commit.
- One change per commit, with before/after numbers in the message, and a test
  that fails without the change.
- After changing `src/hfont/intake.py`, every photographed sheet must still
  read (10, 10, 10), and sheets the change is not aimed at must give
  byte-identical crops.
- Never train or tune on the ten photographed writers; they are the held-out
  test.
- Ask the owner before pushing, deleting, or anything else outward-facing or
  hard to undo.
