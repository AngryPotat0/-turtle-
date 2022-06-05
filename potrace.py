import math
from PIL import Image
import numpy as np

class Point():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return "({a},{b})".format(a=self.y,b=self.x)

class SvgPath():
    def __init__(self,type) -> None:
        self.type = type
        self.data = []
    
    def setData(self,list):
        self.data = list

    def __str__(self) -> str:
        path = self.type
        i = 0
        while(i < len(self.data)):
            if(i != 0): path += ","
            path += str(self.data[i]) + ' ' + str(self.data[i + 1])
            i += 2
        return path

class SVG():
    def __init__(self,height,width) -> None:
        self.height = height
        self.width  = width
        self.pathList = list()
        self.strokec = ""
        self.fillc = ""
        self.fillrule = ""
    
    def appendPath(self,svgPath: SvgPath):
        self.pathList.append(svgPath)
    
    def extendPath(self, svgPathList):
        self.pathList.extend(svgPathList)
    
    def __str__(self) -> str:
        svg = '<svg id="svg" version="1.1" width="' + str(self.width) + '" height="' + str(self.height) + '" xmlns="http://www.w3.org/2000/svg">'
        svg += '<path d="'
        for path in self.pathList:
            svg += str(path) + " "
        svg += ('" stroke="' + self.strokec + '" fill="' + self.fillc + '"' + self.fillrule + '/></svg>')
        return svg

class Path():
    def __init__(self) -> None:
        self.area = 0
        self.lens = 0
        self.curve = dict()
        self.pointList = list()
        self.minX = 100000
        self.minY = 100000
        self.maxX = -1
        self.maxY = -1
        self.sums = None
        self.lon = None
        self.m = 0
        self.po = None
        self.x0 = 0
        self.y0 = 0
        self.curve = None

    def __str__(self) -> str:
        lis = ",".join([str(p) for p in self.pointList])
        return "area:" + str(self.area) + " " + str(self.lens) + " " + lis

class Sum():
    def __init__(self,x,y,xy,x2,y2) -> None:
        self.x = x
        self.y = y
        self.xy = xy
        self.x2  =x2
        self.y2 = y2

class Quad():
    def __init__(self) -> None:
        self.data = [0,0,0,0,0,0,0,0,0]

    def at(self,x,y):
        return self.data[x * 3 + y]

class Curve():
    def __init__(self,n) -> None:
        self.n = n
        self.tag = [0] * n
        self.c = [0 for _ in range(n * 3)]
        self.alphaCurve = 0
        self.vertex = [0] * n
        self.alpha = [0] * n
        self.alpha0 = [0] * n
        self.bate = [0] * n

    def get_c(self):
        return " ".join([str(i) for i in self.c])

def mod(a,n):
    if(a >= n):
        return a % n
    else:
        return a if(a>=0) else (n-1-(-1-a) % n)

def xprod(p1: Point,p2: Point):
    return p1.x * p2.y - p1.y * p2.x

def cyclic(a,b,c):
    if (a <= c):
        return (a <= b and b < c)
    else:
        return (a <= b or b < c)

def sign(i):
    if(i > 0):
        return 1
    else:
        return (-1 if(i < 0) else 0)

def quadform(Q,w):
    v = [0] * 3
    sum = 0,0,0
    v[0] = w.x
    v[1] = w.y
    v[2] = 1
    sum = 0
    for i in range(3):
        for j in range(3):
            sum += v[i] * Q.at(i,i) * v[j]
    return sum

def interval(lbd, a,b):
    res = Point(0,0)
    res.x = a.x + lbd * (b.x - a.x)
    res.y = a.y + lbd * (b.y - a.y)
    return res

def dorth_infty(p0,p2):
    r = Point(0,0)
    r.y = sign(p2.x - p0.x)
    r.x = -sign(p2.y - p0.y)
    return r

def ddenom(p0,p2):
    r = dorth_infty(p0,p2)
    return r.y * (p2.x - p0.x) - r.x * (p2.y - p0.y)

def dpara(p0,p1,p2):
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p2.x - p0.x
    y2 = p2.y - p0.y
    return x1 * y2 - x2 * y1


def get_path(curve: Curve, size):
    fix = lambda a,b: str(round(a,b))
    # print(curve.n,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print(curve.get_c())
    def bezier(i):
        nonlocal curve
        b = 'C' + fix((curve.c[i * 3 + 0].x * size),3) + ' ' + fix((curve.c[i * 3 + 0].y * size),3) + ','
        b += fix((curve.c[i * 3 + 1].x * size),3) + ' ' + fix((curve.c[i * 3 + 1].y * size),3) + ','
        b += fix((curve.c[i * 3 + 2].x * size),3) + ' ' + fix((curve.c[i * 3 + 2].y * size),3) + ' '
        return b
    
    def segment(i):
        nonlocal curve
        nonlocal size
        s = 'L' + fix((curve.c[i * 3 + 1].x * size),3) + ' ' + fix((curve.c[i * 3 + 1].y * size),3) + ' '
        s += fix((curve.c[i * 3 + 2].x * size),3) + ' ' + fix((curve.c[i * 3 + 2].y * size),3) + ' '
        return s
    
    n = curve.n
    p = 'M' + fix((curve.c[(n - 1) * 3 + 2].x * size),3) + ' ' + fix((curve.c[(n - 1) * 3 + 2].y * size),3) + ' '
    for i in range(n):
        if(curve.tag[i] == "CURVE"):
            p += bezier(i)
        elif(curve.tag[i] == "CORNER"):
            p += segment(i)
    return p

def get_path_obj(curve: Curve, size):
    fix = lambda a,b: round(a,b)
    def bezier(i):
        nonlocal curve
        svgPath = SvgPath('C')
        svgPath.setData([fix((curve.c[i * 3 + 0].x * size),3),fix((curve.c[i * 3 + 0].y * size),3),
                        fix((curve.c[i * 3 + 1].x * size),3),fix((curve.c[i * 3 + 1].y * size),3),
                        fix((curve.c[i * 3 + 2].x * size),3),fix((curve.c[i * 3 + 2].y * size),3)])
        return svgPath
    
    def segment(i):
        nonlocal curve
        nonlocal size
        svgPath = SvgPath('L')
        svgPath.setData([fix((curve.c[i * 3 + 1].x * size),3),fix((curve.c[i * 3 + 1].y * size),3),
                        fix((curve.c[i * 3 + 2].x * size),3),fix((curve.c[i * 3 + 2].y * size),3)])
        return svgPath
    
    n = curve.n
    lis = []
    svgPath = SvgPath('M')
    svgPath.setData([fix((curve.c[(n - 1) * 3 + 2].x * size),3),fix((curve.c[(n - 1) * 3 + 2].y * size),3)])
    # print(svgPath)
    lis.append(svgPath)
    for i in range(n):
        if(curve.tag[i] == "CURVE"):
            lis.append(bezier(i))
        elif(curve.tag[i] == "CORNER"):
            lis.append(segment(i))
    return lis


class Potrace():
    def __init__(self,fileName) -> None:
        self.fileName = fileName
        self.img = None
        self.height = 0
        self.width = 0
        self.thresholdArea = 2
        self.max_alpha = 1
        self.pathList = []
    
    def showImg(self):
        for i in range(self.height):
            for j in range(self.width):
                print(self.img[i][j],end=" ")
            print()

    def countImg(self):
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                if(self.img[i][j] == 1): count += 1
        return count
    
    def getPathList(self):
        currentPoint = Point(0,0)

        def findPixel(x,y):
            if(x >= 0 and x < self.width and y >= 0 and y < self.height):
                return self.img[y][x]
            return 0

        def findNextPoint(point):
            j,y = point.x, point.y
            for i in range(y,self.height):
                while(j < self.width):
                    if(self.img[i][j] == 1):
                        return Point(j,i)
                    j += 1
                j = 0
            return None

        def majority(x,y):
            count = 0
            lmd = lambda x: 1 if x else -1
            for i in range(2,5):
                count = 0
                for a in range(-i + 1,i):
                    count += lmd(findPixel(x + a,y + i - 1))
                    count += lmd(findPixel(x + i - 1,y + a - 1))
                    count += lmd(findPixel(x + a - 1,y - i))
                    count += lmd(findPixel(x - i,y + a))
                if(count > 0):
                    return 1
                elif(count < 0):
                    return 0
            return 0

        def findPath(point):
            path = Path()
            x,y = point.x, point.y
            dx,dy = 0,1
            while(True):
                path.pointList.append(Point(x,y))
                if(x > path.maxX): path.maxX = x
                if(x < path.minX): path.minX = x
                if(y > path.maxY): path.maxY = y
                if(y < path.minY): path.minY = y
                path.lens += 1
                x += dx
                y += dy
                path.area -= x * dy
                if(x == point.x and y == point.y): break
                left = findPixel(x + int((dx + dy - 1) / 2), y + int((dy - dx - 1) / 2))
                right = findPixel(x + int((dx - dy - 1) / 2),y + int((dy + dx - 1) / 2))
                #使新的边都有一个黑色像素在它的左边一个白色像素在右边
                if(left == 0 and right == 1):
                    #turn policy: minority
                    if(not majority(x,y)):
                        dx,dy = -dy,dx
                    else:
                        dx,dy = dy, -dx
                elif(right == 1):
                    dx,dy = -dy,dx
                elif(left == 0):
                    dx,dy = dy, -dx
            return path
        
        def reversePath(path):
            lens = path.lens
            y1 = path.pointList[0].y
            maxX, minY = path.maxX,0
            for i in range(1,lens):
                x,y = path.pointList[i].x,path.pointList[i].y
                if(y != y1):
                    minY = y1 if y1 < y else y
                    maxX = path.maxX
                    for j in range(x,maxX):
                        self.img[minY][j] = 0 if self.img[minY][j] == 1 else 1
                    y1 = y
        
        currentPoint = findNextPoint(currentPoint)
        while(currentPoint != None):
            # print("x:",currentPoint.x," y:",currentPoint.y)
            path = findPath(currentPoint)
            # print(path)
            reversePath(path)
            # print("CountImg:",self.countImg())
            if(path.area > self.thresholdArea):
                self.pathList.append(path)
            currentPoint = findNextPoint(currentPoint)
    
    def processPath(self):
        def calcSums(path: Path):
            path.x0,path.y0 = path.pointList[0].x, path.pointList[0].y
            path.sums = []
            path.sums.append(Sum(0,0,0,0,0))
            for i in range(0,path.lens):
                x = path.pointList[i].x - path.x0
                y = path.pointList[i].y - path.y0
                path.sums.append(Sum(path.sums[i].x + x,path.sums[i].y + y,path.sums[i].xy + x * y,path.sums[i].x2 + x * x,path.sums[i].y2 + y * y))
        
        def calcLon(path: Path):
            lens = path.lens
            pointList = path.pointList
            # print("PointList::")
            # for p in pointList:
            #     print(p,end=" ")
            # print()
            dir = 0
            pivk = [0] * lens
            nc = [0] * lens
            ct = None
            path.lon = [0] * lens
            constraint = None
            cur = Point(0,0)
            off = Point(0,0)
            dk = Point(0,0)
            i, j, k1, a, b, c, d, k = 0,0,0,0,0,0,0,0
            for i in range(lens - 1,-1,-1):
                if(pointList[i].x != pointList[k].x and pointList[i].y != pointList[k].y):
                    k = i + 1
                nc[i] = k
            # print("NC::")
            # for cc in nc:
            #     print(cc, end=" ")
            # print()
            for i in range(lens - 1, -1, -1):
                ct = [0,0,0,0]
                dir = (3 + 3 * (pointList[mod(i + 1,lens)].x - pointList[i].x) + (pointList[mod(i + 1,lens)].y - pointList[i].y)) / 2
                ct[int(dir)] += 1
                constraint = [Point(0,0), Point(0,0)]
                k = nc[i]
                k1 = i
                while(1):
                    foundk = 0
                    dir =  (3 + 3 * sign(pointList[k].x - pointList[k1].x) + sign(pointList[k].y - pointList[k1].y)) / 2
                    ct[int(dir)] += 1
                    if(ct[0] and ct[1] and ct[2] and ct[3]):
                        pivk[i] = k1
                        foundk = 1;
                        break
                    cur.x = pointList[k].x - pointList[i].x
                    cur.y = pointList[k].y - pointList[i].y
                    if (xprod(constraint[0], cur) < 0 or xprod(constraint[1], cur) > 0):
                        break
                    if(abs(cur.x) <= 1 and abs(cur.y) <= 1):
                        pass
                    else:
                        off.x = cur.x + (1 if cur.y >= 0 and (cur.y > 0 or cur.x < 0) else -1)
                        off.y = cur.y + (1 if cur.x <= 0 and (cur.x < 0 or cur.y < 0) else -1)
                        if(xprod(constraint[0], off) >= 0):
                            constraint[0].x = off.x
                            constraint[0].y = off.y
                        off.x = cur.x + (1 if(cur.y <= 0 and (cur.y < 0 or cur.x < 0)) else -1)
                        off.y = cur.y + (1 if(cur.x >= 0 and (cur.x > 0 or cur.y < 0)) else -1)
                        if(xprod(constraint[1], off) <= 0):
                            constraint[1].x = off.x
                            constraint[1].y = off.y
                    k1 = k
                    k = nc[k1]
                    if(not cyclic(k, i, k1)): break
                if(foundk == 0):
                    dk.x = sign(pointList[k].x-pointList[k1].x)
                    dk.y = sign(pointList[k].y-pointList[k1].y)
                    cur.x = pointList[k1].x - pointList[i].x
                    cur.y = pointList[k1].y - pointList[i].y
                    a = xprod(constraint[0], cur)
                    b = xprod(constraint[0], dk)
                    c = xprod(constraint[1], cur)
                    d = xprod(constraint[1], dk)
                    j = 10000000
                    if(b < 0):
                        j = math.floor(a / (-b))
                    if(d > 0):
                        j = min(j,math.floor(-c / d))
                    pivk[i] = mod(k1+j,lens)
            
            #FIXME:
            # print("PIVK::")
            # for kk in pivk:
            #     print(kk, end=" ")
            # print()

            j=pivk[lens-1]
            path.lon[lens-1]=j
            for i in range(lens - 2,-1,-1):
                if(cyclic(i+1,pivk[i],j)):
                    j=pivk[i]
                path.lon[i]=j
            ti = lens - 1
            while(cyclic(mod(ti+1,lens),j,path.lon[ti])):
                path.lon[ti] = j
                ti -= 1
            # print("lon::")
            # for i in range(lens):
            #     print(path.lon[i],end=" ")
            # print()

        def bestPolygon(path: Path):
            def penalty3(path: Path,i,j):
                lens = path.lens
                pointList = path.pointList
                sums = path.sums
                r = 0
                x, y, xy, x2, y2 = 0,0,0,0,0
                k, a, b, c, s = 0,0,0,0,0
                px, py, ex, ey = 0,0,0,0
                if(j >= lens):
                    j -= lens
                    r = 1
                if(r == 0):
                    x = sums[j+1].x - sums[i].x
                    y = sums[j+1].y - sums[i].y
                    x2 = sums[j+1].x2 - sums[i].x2
                    xy = sums[j+1].xy - sums[i].xy
                    y2 = sums[j+1].y2 - sums[i].y2
                    k = j + 1 - i
                    # print("a:",j,i,lens)
                else:
                    x = sums[j+1].x - sums[i].x + sums[lens].x
                    y = sums[j+1].y - sums[i].y + sums[lens].y
                    x2 = sums[j+1].x2 - sums[i].x2 + sums[lens].x2
                    xy = sums[j+1].xy - sums[i].xy + sums[lens].xy
                    y2 = sums[j+1].y2 - sums[i].y2 + sums[lens].y2
                    k = j + 1 - i + lens
                    # print("b:",j,i,lens)
                px = (pointList[i].x + pointList[j].x) / 2.0 - pointList[0].x
                py = (pointList[i].y + pointList[j].y) / 2.0 - pointList[0].y
                ey = (pointList[j].x - pointList[i].x)
                ex = -(pointList[j].y - pointList[i].y)
                a = ((x2 - 2*x*px) / k + px*px)
                b = ((xy - x*py - y*px) / k + px*py)
                c = ((y2 - 2*y*py) / k + py*py)
                s = ex*ex*a + 2*ex*ey*b + ey*ey*c
                return math.sqrt(s)

            j,m,k = 0,0,0
            lens = path.lens
            pen = [0] * (lens + 1)
            prev = [0] * (lens + 1)
            clip0 = [0] * (lens)
            clip1 = [0] * (lens + 1)
            seg0 = [0] * (lens + 1)
            seg1 = [0] * (lens + 1)
            thispen, best, c = 0,0,0
            # print("INFOP::")
            # for cvb in path.lon:
            #     print(cvb,end=" ")
            # print()
            for i in range(0,lens):
                c = mod(path.lon[mod(i-1,lens)]-1,lens)
                if(c == i):
                    c = mod(i+1,lens)
                if(c < i):
                    clip0[i] = lens
                else:
                    clip0[i] = c
            j = 1
            for i in range(0,lens):
                while(j <= clip0[i]):
                    clip1[j] = i;
                    j += 1
            i = 0
            j = 0
            # print("clip0::",lens)
            # for yy in clip0:
            #     print(yy,end=" ")
            # print()
            while(i < lens):
                seg0[j] = i
                i = clip0[i];
                j += 1
            seg0[j] = lens
            m = j
            i = lens
            for j in range(m,0,-1):
                seg1[j] = i
                i = clip1[i];
            seg1[0] = 0
            pen[0] = 0
            for j in range(1,m + 1):
                i = seg1[j]
                while(i <= seg0[j]):
                    best = -1
                    k = seg0[j - 1]
                    while(k >= clip1[i]):
                        thispen = penalty3(path,k,i) + pen[k]
                        if(best < 0 or thispen < best):
                            prev[i] = k
                            best = thispen
                        k -= 1
                    pen[i] = best
                    i += 1
            path.m = m
            # print("path_m:", m,lens)
            path.po = [0] * m
            i = lens
            j = m - 1
            while(i > 0):
                i = prev[i]
                path.po[j] = i
                j -= 1

        def adjustVertices(path: Path):
            # print("adj::",path)
            def pointslope(path: Path,i,j,ctr,dir):
                n = path.lens
                sums = path.sums
                k,a,b,c,lambda2,l,r = 0,0,0,0,0,0,0
                while(j >= n):
                    j -= n
                    r += 1
                while(i >= n):
                    i -= n
                    r -= 1
                while(j < 0):
                    j += n
                    r -= 1
                while(i < 0):
                    i += n
                    r += 1
                x = sums[j+1].x-sums[i].x+r*sums[n].x
                y = sums[j+1].y-sums[i].y+r*sums[n].y
                x2 = sums[j+1].x2-sums[i].x2+r*sums[n].x2
                xy = sums[j+1].xy-sums[i].xy+r*sums[n].xy
                y2 = sums[j+1].y2-sums[i].y2+r*sums[n].y2
                k = j+1-i+r*n

                ctr.x = x/k
                ctr.y = y/k

                a = (x2-x*x/k)/k
                b = (xy-x*y/k)/k
                c = (y2-y*y/k)/k

                lambda2 = (a+c+math.sqrt((a-c)*(a-c)+4*b*b))/2
                a -= lambda2
                c -= lambda2

                if(abs(a) >= abs(c)):
                    l = math.sqrt(a*a + b*b)
                    if(l != 0):
                        dir.x = -b / l
                        dir.y = a / l
                else:
                    l = math.sqrt(c*c + b*b)
                    if(l != 0):
                        dir.x = -c / l
                        dir.y = b / l
                if(l == 0):
                    dir.x = dir.y = 0

            m = path.m
            po = path.po
            n = path.lens
            pt = path.pointList
            x0 = path.x0
            y0 = path.y0
            ctr = [Point(0,0) for _ in range(m)]
            dir = [Point(0,0) for _ in range(m)]
            q = [Quad() for _ in range(m)]
            v = [0] * 3
            d,i,j,k,l = 0,0,0,0,0
            s = Point(0,0)
            path.curve = Curve(m)
            for i in range(0,m):
                j = po[mod(i+1,m)]
                j = mod(j-po[i],n)+po[i]
                pointslope(path, po[i], j, ctr[i], dir[i])
            for i in range(0,m):
                d = dir[i].x * dir[i].x + dir[i].y * dir[i].y
                if(d == 0):
                    for j in range(3):
                        for k in range(3):
                            q[i].data[j * 3 + k] = 0
                else:
                    v[0] = dir[i].y
                    v[1] = -dir[i].x
                    v[2] = - v[1] * ctr[i].y - v[0] * ctr[i].x
                    for l in range(3):
                        for k in range(3):
                            q[i].data[l * 3 + k] = v[l] * v[k] / d
            
            Q, w, dx, dy, det, mins, cand, xmin, ymin, z = 0,0,0,0,0,0,0,0,0,0
            for i in range(m):
                Q = Quad()
                w = Point(0,0)
                s.x = pt[po[i]].x-x0
                s.y = pt[po[i]].y-y0
                j = mod(i-1,m)
                # print("i,j,m:",i,j,m)
                for l in range(3):
                    for k in range(3):
                        Q.data[l * 3 + k] = q[j].at(l, k) + q[i].at(l, k)
                # axcv = 0
                while(True):
                    det = Q.at(0, 0)*Q.at(1, 1) - Q.at(0, 1)*Q.at(1, 0)
                    # if(axcv < 6):
                    #     print("det:",det)
                    #     axcv += 1
                    if(det != 0):
                        w.x = (-Q.at(0, 2)*Q.at(1, 1) + Q.at(1, 2)*Q.at(0, 1)) / det
                        w.y = ( Q.at(0, 2)*Q.at(1, 0) - Q.at(1, 2)*Q.at(0, 0)) / det
                        break
                    if(Q.at(0,0) > Q.at(1,1)):
                        v[0] = -Q.at(0, 1)
                        v[1] = Q.at(0, 0)
                    elif(Q.at(1,1)):
                        v[0] = -Q.at(1, 1)
                        v[1] = Q.at(1, 0)
                    else:
                        v[0] = 1
                        v[1] = 0
                    d = v[0] * v[0] + v[1] * v[1]
                    v[2] = - v[1] * s.y - v[0] * s.x
                    for l in range(3):
                        for k in range(3):
                            Q.data[l * 3 + k] += v[l] * v[k] / d
                dx = abs(w.x - s.x)
                dy = abs(w.y - s.y)
                if(dx <= 0.5 and dy <= 0.5):
                    path.curve.vertex[i] = Point(w.x + x0, w.y + y0)
                    continue
                mins = quadform(Q,s)
                xmin, ymin = s.x, s.y
                if(Q.at(0,0) != 0):
                    for z in range(2):
                        w.y = s.y - 0.5 + z
                        w.x = - (Q.at(0, 1) * w.y + Q.at(0, 2)) / Q.at(0, 0)
                        dx = abs(w.x - s.x)
                        cand = quadform(Q,w)
                        if(dx <= 0.5 and cand < mins):
                            mins = cand
                            xmin = w.x
                            ymin = w.y
                if(Q.at(1,1) != 0):
                    for z in range(2):
                        w.x = s.x-0.5+z
                        w.y = - (Q.at(1, 0) * w.x + Q.at(1, 2)) / Q.at(1, 1)
                        dy = abs(w.y-s.y)
                        cand = quadform(Q, w)
                        if(dy < 0.5 and cand < mins):
                            mins = cand
                            xmin = w.x
                            ymin = w.y
                for l in range(2):
                    for k in range(2):
                        w.x = s.x-0.5+l
                        w.y = s.y-0.5+k
                        cand = quadform(Q, w)
                        if(cand < mins):
                            mins = cand
                            xmin = w.x
                            ymin = w.y
                path.curve.vertex[i] = Point(xmin + x0, ymin + y0)
        
        def smooth(path: Path):
            m = path.curve.n
            curve = path.curve
            dd, denom, alpha = 0,0,0
            p2, p3, p4  =0,0,0
            for i in range(m):
                j = mod(i + 1, m)
                k = mod(i + 2, m)
                p4 = interval(1/2.0, curve.vertex[k], curve.vertex[j])
                denom = ddenom(curve.vertex[i], curve.vertex[k])
                if(denom != 0):
                    dd = dpara(curve.vertex[i], curve.vertex[j], curve.vertex[k]) / denom
                    dd = abs(dd)
                    alpha = (1 - 1.0 / dd) if dd > 1 else 0
                    alpha = alpha / 0.75
                else:
                    alpha = 4 / 3.0
                curve.alpha0[j] = alpha
                if(alpha >= self.max_alpha):
                    curve.tag[j] = "CORNER"
                    curve.c[3 * j + 1] = curve.vertex[j]
                    curve.c[3 * j + 2] = p4
                else:
                    if(alpha < 0.55):
                        alpha = 0.55
                    elif(alpha > 1):
                        alpha = 1
                    p2 = interval(0.5+0.5*alpha, curve.vertex[i], curve.vertex[j])
                    p3 = interval(0.5+0.5*alpha, curve.vertex[k], curve.vertex[j])
                    curve.tag[j] = "CURVE"
                    curve.c[3 * j + 0] = p2
                    curve.c[3 * j + 1] = p3
                    curve.c[3 * j + 2] = p4
                curve.alpha[j] = alpha
                curve.bate[j] = 0.5
            curve.alphaCurve = 1
        
        for path in self.pathList:
            calcSums(path)
            calcLon(path)
            bestPolygon(path)
            adjustVertices(path)
            # reverse(path) if path sign == -
            smooth(path)
            #TODO: optiCurve

    def getSVG(self, size, opt_type = ""):
        w = self.width * size
        h = self.height + size
        strokec = ""
        fillc = ""
        fillrule = ""
        svg = '<svg id="svg" version="1.1" width="' + str(w) + '" height="' + str(h) + '" xmlns="http://www.w3.org/2000/svg">'
        svg += '<path d="'
        # print("SVG::",len(self.pathList))
        for i in range(0,len(self.pathList)):
            c = self.pathList[i].curve
            svg += get_path(c,size)
        if(opt_type == "curve"):
            strokec = "black"
            fillc = "none"
            fillrule = ''
        else:
            strokec = "none"
            fillc = "black"
            fillrule = ' fill-rule="evenodd"'
        svg += '" stroke="' + strokec + '" fill="' + fillc + '"' + fillrule + '/></svg>'
        return svg

    def getSvgObj(self, size, opt_type = ""):
        w = self.width * size
        h = self.height + size
        svg = SVG(h,w)
        strokec = ""
        fillc = ""
        fillrule = ""
        # svg = '<svg id="svg" version="1.1" width="' + str(w) + '" height="' + str(h) + '" xmlns="http://www.w3.org/2000/svg">'
        # svg += '<path d="'
        # print("SVG::",len(self.pathList))
        for i in range(0,len(self.pathList)):
            c = self.pathList[i].curve
            # svg += get_path(c,size)
            svg.extendPath(get_path_obj(c,size))
        if(opt_type == "curve"):
            strokec = "black"
            fillc = "none"
            fillrule = ''
        else:
            strokec = "none"
            fillc = "black"
            fillrule = ' fill-rule="evenodd"'
        # svg += '" stroke="' + strokec + '" fill="' + fillc + '"' + fillrule + '/></svg>'
        svg.fillc = fillc
        svg.strokec = strokec
        svg.fillrule = fillrule
        return svg


    def run(self,d_name):
        image = Image.open(self.fileName)
        self.img = np.array(image)
        # print(self.img.shape)
        self.height, self.width, _ = self.img.shape
        bitmap = [[0] * self.width for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if(self.img[i,j,0] == 255 and self.img[i,j,1] == 255 and self.img[i,j,2] == 255):
                    bitmap[i][j] = 0
                else:
                    bitmap[i][j] = 1
        self.img = bitmap
        
        # print(self.height, self.width)
        # self.showImg()
        # print("##################")
        self.getPathList()
        # print(len(self.pathList))
        print("over 1")
        self.processPath()
        print("over 2")
        # svg = self.getSVG(1)
        svg = self.getSvgObj(1)
        if(d_name != ""):
            with open(d_name,"w") as f:
                f.write(str(svg))
        return svg


def main():
    fileName = "result.bmp"
    # fileName = "bmpTest2.bmp"
    test = Potrace(fileName)
    test.run("d.svg")

if __name__ == "__main__":
    main()