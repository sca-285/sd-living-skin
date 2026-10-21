# Third-party notices

Living Skin's own code is MIT (see LICENSE). It uses, without bundling:

- **facexlib** (MIT, https://github.com/xinntao/facexlib), installed by
  `install.py` if missing, for three models it downloads on first use from
  facexlib's GitHub releases into `models/LivingSkin`:
  - RetinaFace (mobilenet 0.25) face detection;
  - BiSeNet face parsing (CelebAMask-HQ labels);
  - MODNet portrait matting.
  The weights are not part of this repository; their own licenses apply.
- **Bebas Neue** (SIL Open Font License 1.1) for the lettering of the preset
  icons; only the rendered pictures are shipped.
- Photos in `preset_icons/` and `reference.jpg`: Open Images dataset, by
  Flickr photographers under CC BY 2.0. Each icon's author and source is in
  `preset_icons/CREDITS.md`; the reference picture names its own.
