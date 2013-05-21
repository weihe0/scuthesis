input TEX;
TEXPRE("%&latex" & char(10) & "\documentclass{article}\usepackage{amsmath,amssymb}\begin{document}");
TEXPOST("\end{document}");
prologues := 3;
u := 0.5cm;
def drawgrid(expr n)=
  pickup pencircle scaled 1pt;
  for i=0 upto n:
    draw (0*u,i*u)--(n*u,i*u);
  endfor;
  for i=0 upto n:
    draw (i*u,0*u)--(i*u,n*u);
  endfor;
enddef;

def putnumber(expr i, j, a)=
  label(TEX decimal(a),(j*u+0.5u,i*u+0.5u));
enddef;

outputtemplate := "box.eps";
beginfig(1);
  drawgrid(3);
  for i=0 upto 2:
    for j=0 upto 2:
      putnumber(i, j, 1);
    endfor;
  endfor;
  label.lft(TEX("$\frac{1}{9}\times$"),(0*u,1.5*u));
endfig;

outputtemplate := "gaussian.eps";
beginfig(2);
  drawgrid(3);
  numeric a[];
  a[0] := 1;
  a[1] := 2;
  a[2] := 1;
  for i=0 upto 2:
    for j=0 upto 2:
      putnumber(i, j, a[i]*a[j]);
    endfor;
  endfor;
  label.lft(TEX("$\frac{1}{16}\times$"),(0*u,1.5*u));
endfig;

outputtemplate := "prewitt_x.eps";
beginfig(3);
  drawgrid(3);
  for i=0 upto 2:
    putnumber(i, 0, -1);
    putnumber(i, 2, 1);
  endfor;
  label.lft(TEX("$\frac{1}{6}\times$"),(0*u,1.5*u));
endfig;

outputtemplate := "prewitt_y.eps";
beginfig(4);
  drawgrid(3);
  for j=0 upto 2:
    putnumber(0, j, -1);
    putnumber(2, j, 1);
  endfor;
  label.lft(TEX("$\frac{1}{6}\times$"),(0*u,1.5*u));
endfig;

outputtemplate := "sobel_x.eps";
beginfig(5);
  drawgrid(3);
  putnumber(0, 0, -1);
  putnumber(0, 2, 1);
  putnumber(1, 0, -2);
  putnumber(1, 2, 2);
  putnumber(2, 0, -1);
  putnumber(2, 2, 1);
  label.lft(TEX("$\frac{1}{8}\times$"),(0*u,1.5*u));
endfig;

outputtemplate := "sobel_y.eps";
beginfig(6);
  drawgrid(3);
  putnumber(0, 0, -1);
  putnumber(0, 1, -2);
  putnumber(0, 2, -1);
  putnumber(2, 0, 1);
  putnumber(2, 1, 2);
  putnumber(2, 2, 1);
  label.lft(TEX("$\frac{1}{8}\times$"),(0*u,1.5*u));
endfig;

outputtemplate := "robert_x.eps";
beginfig(7);
  drawgrid(2);
  putnumber(0, 0, -1);
  putnumber(1, 1, 1);
endfig;

outputtemplate := "robert_y.eps";
beginfig(8);
  drawgrid(2);
  putnumber(0, 1, -1);
  putnumber(1, 0, 1);
endfig;


outputtemplate := "binary.eps";
beginfig(8);
  drawgrid(5);
  for i=1 upto 3:
    for j=2 upto 3:
      putnumber(i, j, 1);
    endfor;
  endfor;
  putnumber(2, 1, 1);
endfig;

outputtemplate := "struct.eps";
beginfig(9);
  drawgrid(3);
  putnumber(0, 1, 1);
  for j=0 upto 2:
    putnumber(1, j, 1);
  endfor;
  putnumber(2, 1, 1);
endfig;

outputtemplate := "dilate.eps";
beginfig(10);
  drawgrid(5);
  for i=0 upto 4:
    for j=2 upto 3:
      putnumber(i, j, 1);
    endfor;
  endfor;
  for i=1 upto 3:
    putnumber(i, 1, 1);
    putnumber(i, 4, 1);
  endfor;
  putnumber(2, 0, 1);
endfig;

outputtemplate := "erode.eps";
beginfig(11);
  drawgrid(5);
  putnumber(2, 2, 1);
endfig;

outputtemplate := "open.eps";
beginfig(12);
  drawgrid(5);
  putnumber(2, 2, 1);
  putnumber(1, 2, 1);
  putnumber(3, 2, 1);
  putnumber(2, 1, 1);
  putnumber(2, 3, 1);
endfig;

outputtemplate := "close.eps";
beginfig(13);
  drawgrid(5);
  for i=1 upto 3:
    putnumber(i, 2 ,1);
    putnumber(i, 3, 1);
  endfor;
  putnumber(2, 1, 1);
endfig;

outputtemplate := "hilditchmask.eps";
beginfig(14);
  drawgrid(3);
  label(TEX("$p$"),(1*u+0.5u,1*u+0.5u));
  label(TEX("$x_1$"),(2*u+0.5u,1*u+0.5u));
  label(TEX("$x_2$"),(2*u+0.5u,2*u+0.5u));
  label(TEX("$x_3$"),(1*u+0.5u,2*u+0.5u));
  label(TEX("$x_4$"),(0*u+0.5u,2*u+0.5u));
  label(TEX("$x_5$"),(0*u+0.5u,1*u+0.5u));
  label(TEX("$x_6$"),(0*u+0.5u,0*u+0.5u));
  label(TEX("$x_7$"),(1*u+0.5u,0*u+0.5u));
  label(TEX("$x_8$"),(2*u+0.5u,0*u+0.5u));
endfig;