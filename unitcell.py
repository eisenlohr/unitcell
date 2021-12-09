#!/usr/bin/env python3

import subprocess
import numpy as np
import argparse

from pathlib import Path

def listify(what,glue=','):
    return glue.join(list(map(str,what)))

unitcell_choices = ['cubic','hexagonal','tetragonal','orthorhombic']

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description = """
Generate vector-based crystal unitcell overlays from Euler angles.

Angles are specified either directly as argument, resulting in a
single file with name derived from the orientation and lattice,
or by scanning through an EDAX/TSL unitcell file from which a
batch of unitcells located at the respective positions is produced.

Requires a working setup of 'sketch' (by Gene Ressler)
and 'LaTeX' with 'TikZ' extension.
""")
geom = parser.add_argument_group('Geometry')
geom.add_argument('--type', default='cubic',
                  help=f'type of unitcell {{{listify(unitcell_choices)}}}')
geom.add_argument('--eulers', nargs=3, type=float, default=[0,0,0], metavar=('phi1','Phi','phi2'),
                  help='3-1-3 Euler angles')
geom.add_argument('--radians', action='store_true',
                  help='Euler angles are in radians')
geom.add_argument('-c', type=float, metavar='c/a',
                  help='unitcell c/a ratio')
geom.add_argument('-b', type=float, metavar='b/a',
                  help='unitcell b/a ratio')
plotting = parser.add_argument_group('Plotting')
plotting.add_argument('--eye', nargs=3, type=float, default=[0,0,1], metavar=('x','y','z'),
                      help='position of eye on scene')
plotting.add_argument('--up', nargs=3, type=float, default=[-1,0,0], metavar=('x','y','z'),
                      help='vector corresponding to up direction')
plotting.add_argument('--perspective', action='store_true',
                      help='use perspective projection')
plotting.add_argument('--opacity', type=float, default=0.8,
                      help='opacity level')
plotting.add_argument('--axes', action='store_true',
                      help='show both (global and crystal) coordinate frames')
plotting.add_argument('--globalaxes', action='store_true',
                      help='show global coordinate frame')
plotting.add_argument('--crystalaxes', action='store_true',
                      help='show crystal coordinate frame')
plotting.add_argument('--globalvector', nargs=3, type=float, metavar=('x','y','z'),
                      help='draw vector in lab frame')
plotting.add_argument('--crystalvector', nargs=3, type=float, metavar=('a','b','c'),
                      help='draw vector in crystal frame')
figure = parser.add_argument_group('Figure')
figure.add_argument('--name',
                    help='output file name')
figure.add_argument('--latex', action='store_true',
                    help='produce LaTeX figure instead of PDF')
figure.add_argument('--keep', action='store_true',
                    help='keep intermediate files')
figure.add_argument('--verbose', action='store_true',
                    help='verbose output')
batch = parser.add_argument_group('Batch processing')
batch.add_argument('--batch', metavar='FILE',
                   help='EDAX/TSL unitcell file')
batch.add_argument('--scale', type=float, default=7.,
                   help='scale of diagonal bounding box in batch file')
batch.add_argument('--label', action='store_true',
                   help='mark batch processed unitcells by number')


args = parser.parse_args()

args.eulers = np.array(args.eulers)
args.up = np.array(args.up)
args.eye = np.array(args.eye)
if args.batch is not None: args.batch = Path(args.batch)

if args.type not in unitcell_choices:
  parser.error(f'"{args.type}" not a valid choice for unitcell [{listify(unitcell_choices)}].')

if np.linalg.norm(np.cross(args.eye,args.up)) < 1.0e-10:
  parser.error('Eye position and up vector cannot be collinear.')

if args.axes: args.globalaxes = args.crystalaxes = args.axes

if args.c is None: args.c = {'hexagonal':1.633}.get(args.type,1)
if args.b is None: args.b = {                 }.get(args.type,1)

opacity = np.clip(args.opacity,0,1)

args.perspective = 'view(({eye[0]},{eye[1]},{eye[2]}),(0,0,0),[{up[0]},{up[1]},{up[2]}]) then perspective({scale})'\
                      .format(eye   = (5+args.scale)*args.eye,
                              up    = args.up,
                              scale = 80.0/args.scale**0.4) if args.perspective else \
                      'view(({eye[0]},{eye[1]},{eye[2]}),(0,0,0),[{up[0]},{up[1]},{up[2]}]) then scale({scale})'\
                      .format(eye   =args.scale*args.eye,
                              up    = args.up,
                              scale = 20.0/args.scale)

coords = {
          'x':[4,1],         # column 4 positive
          'y':[3,1],         # column 3 positive
         }

header = """
def O (0,0,0)
def J [0,0,1]

def px [1,0,0]+(O)
def py [0,1,0]+(O)
def pz [0,0,1]+(O)

def globalSystemAxes
{{

line[color=black,line width=0.5pt] (O)(px)
line[color=black,line width=0.5pt] (O)(py)
line[color=black,line width=0.5pt] (O)(pz)
special |
 \path #1 -- #2 node[color=black,font=\\footnotesize,pos=1.25] {{$x$}};
 \path #1 -- #3 node[color=black,font=\\footnotesize,pos=1.25] {{$y$}};
 \path #1 -- #4 node[color=black,font=\\footnotesize,pos=1.25] {{$z$}};
| (O)(px)(py)(pz)

}}


def localSystemAxes
{{

line[color=red,line width=1.5pt] (O)(px)
line[color=red,line width=1.5pt] (O)(py)
line[color=red,line width=1.5pt] (O)(pz)
special |
 \path #1 -- #2 node[color=red,font=\\footnotesize,pos=1.25] {{$a$}};
 \path #1 -- #3 node[color=red,font=\\footnotesize,pos=1.25] {{$b$}};
 \path #1 -- #4 node[color=red,font=\\footnotesize,pos=1.25] {{$c$}};
| (O)(px)(py)(pz)

}}


def unitcell_hexagonal
{{
  def n 6
  def ca {ca}

  sweep[cull=false,line width={linewidth}pt,fill=white,draw=black,fill opacity={opacity}]
        {{n<>,rotate(360/n,(O),[J])}} line (1,0,-ca/2)(1,0,ca/2)
}}


def unitcell_cubic
{{
  def n 4

  sweep[cull=false,line width={linewidth}pt,fill=white,draw=black,fill opacity={opacity}]
        {{n<>,rotate(360/n,(O),[J])}} line (0.5,0.5,-0.5)(0.5,0.5,0.5)
}}


def unitcell_tetragonal
{{
  def ca {ca}

  put {{ scale([1,1,ca]) }} {{unitcell_cubic}}
}}


def unitcell_orthorhombic
{{
  def ba {ba}
  def ca {ca}

  put {{ scale([1,ba,ca]) }} {{unitcell_cubic}}
}}


"""

rotation = """
def EulerRotation_{0}_{1}_{2}
                        rotate({0},[0,0,1]) then
             rotate({1},rotate({0},[0,0,1])*[1,0,0]) then
  rotate({2},rotate({1},rotate({0},[0,0,1])*[1,0,0])*[0,0,1])

"""

unitcell      = 'put {{ [[EulerRotation_{euler[0]}_{euler[1]}_{euler[2]}]] then translate([{d[0]},{d[1]},0]) }} {{unitcell_{celltype}}}'
latticevector = 'put {{ [[EulerRotation_{euler[0]}_{euler[1]}_{euler[2]}]] then translate([{d[0]},{d[1]},0]) }} {{ line[color=blue ,line width=3pt] (O)({vector}) }}'
labvector     = 'put {{                                                         translate([{d[0]},{d[1]},0]) }} {{ line[color=green,line width=3pt] (O)({vector}) }}'
localaxes     = 'put {{ [[EulerRotation_{euler[0]}_{euler[1]}_{euler[2]}]] then translate([{d[0]},{d[1]},0]) then scale(1.25) }} {{ {{localSystemAxes}} }}'
globalaxes    = '{globalSystemAxes}'
label         = 'special |\\node at #1 {{{num}}};|({d[0]},{d[1]},0)'

view = """
put {{ {how} }}
    {{
     {what}
    }}
"""

footer = """
global { language tikz }
"""


if args.batch is None or not args.batch.exists():
  args.eulers = np.degrees(args.eulers)%360 if args.radians else args.eulers%360
  eulerInts = np.round(args.eulers).astype(int)

  cmd = header.format(ca=args.c,
                      ba=args.b,
                      linewidth=40/args.scale,
                      opacity=args.opacity,
                     )
  cmd += rotation.format(*eulerInts)
  cmd += view.format(how=args.perspective,
                     what=(globalaxes if args.globalaxes else '') +
                          (localaxes.format(euler = eulerInts,
                                            d = [0,0],
                                           ) if args.crystalaxes  else '') +
                          (latticevector.format(euler = eulerInts,
                                                d = [0,0],
                                                vector = listify(args.crystalvector),
                                               ) if args.crystalvector else '') +
                          (labvector.format(d = [0,0],
                                            vector = listify(args.globalvector),
                                           ) if args.globalvector  else '') +
                          unitcell.format(euler = eulerInts,
                                          d = [0,0],
                                          celltype = args.type,
                                          ),
                     )
  cmd += footer

  filename = Path(args.name if args.name else 'unitcell_{0}_{1}_{2}_{3}'.format(args.type,*eulerInts))

else:

  content = [line for line in args.batch.read_text() if line.find('#') != 0]
  offset = int(args.batch.suffix.lower() != '.ang')
  dataset = np.array([map(float,line.split(None if content[0].find(',') < 0 else ',')[offset:offset+5]) for line in content])
  dataset[:,[coords['x'][0],coords['x'][0]]] *= np.array([coords['x'][1],coords['y'][1]])
  if args.radians: dataset[:,:3] = np.degrees(dataset[:,:3])

  boundingBox = np.array([np.min(dataset[:,[coords['x'][0],coords['y'][0]]],axis=0),
                          np.max(dataset[:,[coords['x'][0],coords['y'][0]]],axis=0),
                         ])
  centre = np.average(boundingBox,axis=0)
  scale = args.scale / np.linalg.norm(boundingBox[1]-boundingBox[0])

  rotations = {}
  cells = []
  labels = []

  counter = 0

  for point in dataset:
    counter += 1
    x = coords['x'][1]*point[coords['x'][0]]
    y = coords['y'][1]*point[coords['y'][0]]

    eulerInts = np.round(point[:3]).astype(int)
    rotations[listify(eulerInts,glue='#')] = rotation.format(*eulerInts)
    cells.append(unitcell.format(euler=eulerInts,
                                 d=scale*(point[[coords['x'][0],coords['x'][0]]]-centre),
                                 celltype=args.type)
                )
    if args.label:
      labels.append(label.format(num=counter,
                                 d=scale*(point[[coords['x'][0],coords['x'][0]]]-centre),
                                 ))

  cmd = header.format(ca=args.c,
                      ba=args.b,
                      linewidth=80/args.scale,
                      opacity=args.opacity
                     )
  cmd += '\n'.join(rotations.values())
  cmd += view.format(how=args.perspective,
                     what=args.axes + '\n'.join(cells) + '\n'.join(labels),
                   )
  cmd += footer

  filename = Path(f'unitcell_{args.type}_{args.batch.stem}')

filename.with_suffix('.sk').write_text(cmd)

if args.latex:
    subprocess.run(['sketch',
                    '-o',f'{filename.with_suffix(".tex")}',
                    f'{filename.with_suffix(".sk")}',
                    ],stdout=None if args.verbose else subprocess.DEVNULL,stderr=subprocess.STDOUT)
    if not args.keep:
        filename.with_suffix('.sk').unlink()
else:
    subprocess.run(['sketch',
                    '-Tb',
                    '-o',f'{filename.with_suffix(".tex")}',
                    f'{filename.with_suffix(".sk")}',
                    ],stdout=None if args.verbose else subprocess.DEVNULL,stderr=subprocess.STDOUT)
    subprocess.run(['pdflatex',
                    '-output-directory',f'{filename.parent}',
                    f'{filename.with_suffix(".tex")}',
                    ],stdout=None if args.verbose else subprocess.DEVNULL,stderr=subprocess.STDOUT)
    for ext in [] if args.keep else ['.sk','.tex','.log','.aux']:
        filename.with_suffix(ext).unlink()
