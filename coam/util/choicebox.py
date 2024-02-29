import sys
import tkinter
from tkinter import ttk


# Based on https://stackoverflow.com/a/10065345
class ChoiceBox:
    def __init__(self, msg, choices, b1, b2, frame, t, entry):
        root = self.root = tkinter.Tk()
        root.title("COAM")
        self.msg = str(msg)
        self.returning = choices[0]
        if not frame:
            root.overrideredirect(True)

        frm_1 = tkinter.Frame(root)
        frm_1.pack(ipadx=2, ipady=2)
        message = tkinter.Label(frm_1, text=self.msg)
        message.pack(padx=8, pady=8)
        self.c = ttk.Combobox(frm_1, value=choices if choices else [], state="readonly")
        self.c.set(choices[0])
        self.c.bind("<<ComboboxSelected>>", self.combobox_select)
        self.c.pack(padx=8, pady=8)

        if entry:
            self.entry = tkinter.Entry(frm_1)
            self.entry.pack()
            self.entry.focus_set()

        frm_2 = tkinter.Frame(frm_1)
        frm_2.pack(padx=4, pady=4)

        btn_1 = tkinter.Button(frm_2, width=8, text=b1)
        btn_1["command"] = self.b1_action
        root.bind("<Return>", self.b1_action)
        btn_1.focus_set()
        btn_1.pack()
        if not entry:
            btn_1.focus_set()

        btn_1.bind("<KeyPress-Return>", func=self.b1_action)
        root.update_idletasks()
        xp = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        yp = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        geom = (root.winfo_width(), root.winfo_height(), xp, yp)
        root.geometry("{}x{}+{}+{}".format(*geom))

        root.deiconify()
        root.protocol("WM_DELETE_WINDOW", self.close_mod)

    def combobox_select(self, event):
        self.returning = self.c.get()
        print(self.returning)

    def close_mod(self):
        sys.exit(0)

    def b1_action(self, event=None):
        self.root.quit()


def choicebox(msg, choices, b1="OK", b2="Cancel", frame=True, t=False, entry=False):
    """Create an instance of MessageBox, and get data back from the user.
    msg = string to be displayed
    b1 = text for left button, or a tuple (<text for button>, <to return on press>)
    b2 = text for right button, or a tuple (<text for button>, <to return on press>)
    frame = include a standard outerframe: True or False
    t = time in seconds (int or float) until the msgbox automatically closes
    entry = include an entry widget that will have its contents returned: True or False
    """
    msgbox = ChoiceBox(msg, choices, b1, b2, frame, t, entry)
    msgbox.root.mainloop()
    # the function pauses here until the mainloop is quit
    msgbox.root.destroy()
    return msgbox.returning
