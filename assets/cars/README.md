# Car sprite placeholders

The PNGs in this directory are reproducibly extracted from
`sprites/spriteCarros.jpg` with:

```bash
python tools/extract_car_sprites.py
```

The development tool requires Pillow (`python -m pip install Pillow`). Python
is not used by the application at runtime. Run the command with `--check` to
validate that all 11 outputs are RGBA images on a 280x150 canvas. Car 06 is
intentionally omitted because its source sprite is defective; Car 03 replaces
it in the ten-car training population.

The visible vehicle artwork fills the normalized canvas so the rendered sprite
matches the fixed 28x15 physical hitbox.

The supplied source visibly contains stock-image branding/watermarking. The
script does not remove or conceal it; it only crops the vehicle regions and
removes their plain connected background. Treat these images as development
placeholders and use appropriately licensed replacement sprites before public
or production distribution when necessary.
