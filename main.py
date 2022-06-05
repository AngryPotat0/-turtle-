from PIL import Image
from statistics import mean
import numpy as np
from potrace import*

code = []
section = []
section.append("import turtle as te\n")

code.append("WriteStep = 15\n")
Speed = 1000
Width = 600  # 界面宽度
Height = 600  # 界面高度
scale = (1, 1)
first = True
K = 32

func = """
def Bezier(p1, p2, t):
    return p1 * (1 - t) + p2 * t

def Bezier_2(x1, y1, x2, y2, x3, y3):
    te.goto(x1, y1)
    te.pendown()
    for t in range(0, WriteStep + 1):
        x = Bezier(Bezier(x1, x2, t / WriteStep),
                    Bezier(x2, x3, t / WriteStep), t / WriteStep)
        y = Bezier(Bezier(y1, y2, t / WriteStep),
                    Bezier(y2, y3, t / WriteStep), t / WriteStep)
        te.goto(x, y)
    te.penup()


def Bezier_3(x1, y1, x2, y2, x3, y3, x4, y4):
    x1 = -Width / 2 + x1
    y1 = Height / 2 - y1
    x2 = -Width / 2 + x2
    y2 = Height / 2 - y2
    x3 = -Width / 2 + x3
    y3 = Height / 2 - y3
    x4 = -Width / 2 + x4
    y4 = Height / 2 - y4
    te.goto(x1, y1)
    te.pendown()
    for t in range(0, WriteStep + 1):
        x = Bezier(Bezier(Bezier(x1, x2, t / WriteStep), Bezier(x2, x3, t / WriteStep), t / WriteStep),
                    Bezier(Bezier(x2, x3, t / WriteStep), Bezier(x3, x4, t / WriteStep), t / WriteStep), t / WriteStep)
        y = Bezier(Bezier(Bezier(y1, y2, t / WriteStep), Bezier(y2, y3, t / WriteStep), t / WriteStep),
                    Bezier(Bezier(y2, y3, t / WriteStep), Bezier(y3, y4, t / WriteStep), t / WriteStep), t / WriteStep)
        te.goto(x, y)
    te.penup()


def Moveto(x, y):
    te.penup()
    te.goto(-Width / 2 + x, Height / 2 - y)
    te.pendown()

def line(x1, y1, x2, y2):
    te.penup()
    te.goto(-Width / 2 + x1, Height / 2 - y1)
    te.pendown()
    te.goto(-Width / 2 + x2, Height / 2 - y2)
    te.penup()

def Lineto(x, y):
    te.pendown()
    te.goto(-Width / 2 + x, Height / 2 - y)
    te.penup()

def Curveto(x1, y1, x2, y2, x, y):
    te.penup()
    X_now = te.xcor() + Width / 2
    Y_now = Height / 2 - te.ycor()
    Bezier_3(X_now, Y_now, x1, y1, x2, y2, x, y)
    
\n"""
code.append(func)



def genCode(svg: SVG, w_color):
    global first
    Height = svg.height
    Width = svg.width
    if first:
        section.append("Width={w}\nHeight={h}\n".format(w=Width,h=Height))
        section.append("te.setup(width=Width, height=Height)\n")
        # section.append("""te.setworldcoordinates(-Width / 2, Height / 2, Width - Width / 2, -Height + Height / 2)\n""")
        first = False
    code.extend(["te.tracer(100)\n","te.pensize(1)\n","te.speed({s})\n".format(s=Speed),"te.penup()\n","te.color(\"{c}\")\n".format(c=w_color)])
    for i in svg.pathList:
        if i.type == 'M':
            code.append("te.end_fill()\n")
            code.append("Moveto({a} * {b}, {c} * {d})\n".format(a=i.data[0],b=1,c=i.data[1],d=1))
            code.append("te.begin_fill()\n")
        elif i.type == 'C':
            cd = """Curveto({a},{b},{c},{d},{e},{f})\n""".format(a=i.data[0],b=i.data[1],c=i.data[2],d=i.data[3],e=i.data[4],f=i.data[5])
            code.append(cd)
        elif i.type == 'L':
            # code.append("Lineto({a},{b})\n".format(a=i.data[0],b=i.data[1]))
            code.append("line({a},{b},{c},{d})\n".format(a=i.data[0],b=i.data[1],c=i.data[2],d=i.data[3]))
    code.extend(["te.penup()\n","te.hideturtle()\n","te.update()\n"])



class Cube(object):
    def __init__(self, colors):
        self.colors = colors or []
        self.red = [r[0] for r in colors]
        self.green = [g[1] for g in colors]
        self.blue = [b[2] for b in colors]
        self.size = (max(self.red) - min(self.red),
                        max(self.green) - min(self.green),
                        max(self.blue) - min(self.blue))
        self.range = max(self.size)
        self.channel = self.size.index(self.range)

    def __lt__(self, other):
        return self.range < other.range

    def average(self):
        r = int(mean(self.red))
        g = int(mean(self.green))
        b = int(mean(self.blue))
        return r, g, b
        
    def split(self):
        middle = int(len(self.colors) / 2)
        colors = sorted(self.colors, key=lambda c: c[self.channel])
        return Cube(colors[:middle]), Cube(colors[middle:])

def median_cut(img,height,width,num):#中位切分法减色
    colors = []
    # for count, color in img.getcolors(img.width * img.height):
    #     colors += [color]
    for i in range(height):
        for j in range(width):
            colors.append(img[i,j])

    cubes = [Cube(colors)]

    while len(cubes) < num:
        cubes.sort()
        cubes += cubes.pop().split()
    i = 0
    LUT = {}
    index = ()
    for c in cubes:
        average = c.average()
        for color in c.colors:
            LUT[(color[0]),color[1],color[2]] = average                
    return LUT

def getColorTable(lis):#获取颜色表
    colorTable = set()
    for k in lis.keys():
        colorTable.add(lis[k])
    # print("ColorTable:",colorTable)
    return colorTable

def getColorTableFormImg(img,height,width): #直接从图像中获取颜色表，测试用，正式使用中无用
    colorTable = set()
    for i in range(height):
        for j in range(width):
            color = img[i,j]
            colorTable.add((color[0],color[1],color[2]))
    return colorTable

def splitImg(img,img_height,img_width,colorTable):
    color_to_img = {}
    split_img_list = []
    lis = [[(255,255,255)] * img_width for _ in range(img_height)]
    for color in colorTable:
        color_to_img[color] = np.array(lis,dtype=int)

    for i in range(img_height):
        for j in range(img_width):
            color = img[i,j]
            color_to_img[(color[0],color[1],color[2])][i,j] = (0,0,0)
    for color in color_to_img.keys():
        split_img_list.append((color,color_to_img[color]))
    return split_img_list

def dealMedian(img,height,width):#中值滤波
    dest = [[0,0,0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            if(i <= 0 or i >= height - 1 or j <= 0 or j >= width - 1):
                dest[i][j] = img[i,j]
            else:
                lisR = sorted([img[i-1,j-1,0],img[i-1,j,0],img[i-1,j+1,0],img[i,j-1,0],img[i,j,0],img[i,j+1,0],img[i+1,j-1,0],img[i+1,j,0],img[i+1,j+1,0]])
                lisG = sorted([img[i-1,j-1,1],img[i-1,j,1],img[i-1,j+1,1],img[i,j-1,1],img[i,j,1],img[i,j+1,1],img[i+1,j-1,1],img[i+1,j,1],img[i+1,j+1,1]])
                lisB = sorted([img[i-1,j-1,2],img[i-1,j,2],img[i-1,j+1,2],img[i,j-1,2],img[i,j,2],img[i,j+1,2],img[i+1,j-1,2],img[i+1,j,2],img[i+1,j+1,2]])
                dest[i][j] = [lisR[4],lisG[4],lisB[4]]
    for i in range(height):
        for j in range(width):
            img[i,j] = dest[i][j]

def main() :
    imgName = input("input Img name:")
    level = int(input("split level:"))
    image = Image.open(imgName)
    img = np.array(image)
    rows,cols,channel=img.shape
    dealMedian(img,rows,cols)
    print("中值滤波结束")
    LUT = median_cut(img,rows,cols,level)
    colorTable = getColorTable(LUT)
    for i in range(rows):
        for j in range(cols):
            index = (img[i,j,0],img[i,j,1],img[i,j,2])
            color = LUT[index]
            img[i,j] = color
    print("量化减色结束")
    resImg = Image.fromarray(img)
    resImg.save("res.bmp","BMP")
    split_imgs = splitImg(img,rows,cols,colorTable)
    print("分层完成，开始生成turtle程序,共计{i}层".format(i=len(split_imgs)))
    count = 0
    for imgs in split_imgs:
        count += 1
        print("开始生成第{i}层".format(i = count))
        color,aimg = imgs[0],imgs[1]
        timg = Image.fromarray(np.uint8(aimg))
        timg.save("result.bmp","BMP")
        # os.system('potrace result.bmp -s --flat')
        p = Potrace("result.bmp")
        svg = p.run("result.svg")
        genCode(svg, '#%02x%02x%02x' % (color[0],color[1],color[2]))
    print("生成完成")
    code.append("""print("Done")\n""")
    code.append("te.done()\n")
    # print("########################")
    # print("".join(code))
    with open("code.py","w") as f:
        for c in section:
            f.writelines(c)
        for c in code:
            f.writelines(c)


if __name__ == "__main__":
    main()