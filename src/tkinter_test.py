
from Tkinter import *
import time

root = Tk()
root.title('very')
canvas_1 = Canvas(root,width=300,height=200,background='#ffffff')
#canvas_1.grid(row=2,column=2)

#canvas_1.create_line(10,20,50,70)

#canvas_1.update_idletasks()
canvas_1.pack(expand=True, fill=BOTH)



x = 10

def update_fnt():
    for i in range(0,10):
        canvas_1.delete(ALL)
        global x
        incr = 20
        x = x + incr
        print 'hello'
        canvas_1.create_line(x,20,50,70)
        
        canvas_1.update()
        time.sleep(1)
        
    root.quit()
        
        
        
    
canvas_1.after(2, update_fnt)

root.mainloop()


#http://82.157.70.109/mirrorbooks/pythonprogramming/0596000855_python2-CHP-8-SECT-7.html
'''
from Tkinter import *
trace = 0 

class CanvasEventsDemo1: 
    def __init__(self, parent=None):
        canvas = Canvas(width=300, height=300, bg='beige') 
        canvas.pack()
        canvas.bind('<ButtonPress-1>', self.onStart)      # click
        canvas.bind('<B1-Motion>',     self.onGrow)       # and drag
        canvas.bind('<Double-1>',      self.onClear)      # delete all
        canvas.bind('<ButtonPress-3>', self.onMove)       # move latest
        self.canvas = canvas
        self.drawn  = None
        self.kinds = [canvas.create_oval, canvas.create_rectangle]
    def onStart(self, event):
        self.shape = self.kinds[0]
        self.kinds = self.kinds[1:] + self.kinds[:1]      # start dragout
        self.start = event
        self.drawn = None
    def onGrow(self, event):                              # delete and redraw
        canvas = event.widget
        if self.drawn: canvas.delete(self.drawn)
        objectId = self.shape(self.start.x, self.start.y, event.x, event.y)
        if trace: print objectId
        self.drawn = objectId
    def onClear(self, event):
        event.widget.delete('all')                        # use tag all
    def onMove(self, event):
        if self.drawn:                                    # move to click spot
            if trace: print self.drawn
            canvas = event.widget
            diffX, diffY = (event.x - self.start.x), (event.y - self.start.y)
            canvas.move(self.drawn, diffX, diffY)
            self.start = event

#if __name__ == '__main__':
#    CanvasEventsDemo()
#    mainloop()



from Tkinter import *
import time

class CanvasEventsDemo(CanvasEventsDemo1):
    def __init__(self, parent=None):
        CanvasEventsDemo1.__init__(self, parent)
        self.canvas.create_text(75, 8, text='Press o and r to move shapes')
        self.canvas.master.bind('<KeyPress-o>', self.onMoveOvals)    
        self.canvas.master.bind('<KeyPress-r>', self.onMoveRectangles)  
        self.kinds = self.create_oval_tagged, self.create_rectangle_tagged
    def create_oval_tagged(self, x1, y1, x2, y2):
        objectId = self.canvas.create_oval(x1, y1, x2, y2)
        self.canvas.itemconfig(objectId, tag='ovals', fill='blue')
        return objectId
    def create_rectangle_tagged(self, x1, y1, x2, y2):
        objectId = self.canvas.create_rectangle(x1, y1, x2, y2)
        self.canvas.itemconfig(objectId, tag='rectangles', fill='red')
        return objectId
    def onMoveOvals(self, event):
        print 'moving ovals'
        self.moveInSquares(tag='ovals')           # move all tagged ovals
    def onMoveRectangles(self, event):
        print 'moving rectangles'
        self.moveInSquares(tag='rectangles')
    def moveInSquares(self, tag):                 # 5 reps of 4 times per sec
        for i in range(5):
            for (diffx, diffy) in [(+20, 0), (0, +20), (-20, 0), (0, -20)]:
                self.canvas.move(tag, diffx, diffy)
                self.canvas.update()              # force screen redraw/update
                time.sleep(0.25)                  # pause, but don't block gui

if __name__ == '__main__':
    CanvasEventsDemo()
    mainloop()
'''