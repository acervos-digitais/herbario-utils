def save_crop(img, box, fpath, max_height=512):
  (x0,y0,x1,y1) = box
  iw,ih = img.size
  bimg = img.crop((x0*iw, y0*ih, x1*iw, y1*ih))
  biw, bih = bimg.size
  if bih > max_height:
    bimg = bimg.resize((int(biw * max_height / bih), max_height))
  bimg.save(fpath)
