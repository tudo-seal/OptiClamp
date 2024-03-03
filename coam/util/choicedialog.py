import os

import customtkinter as ctk


# Based on https://stackoverflow.com/a/10065345
class CTkChoiceDialog(ctk.CTkToplevel):
    def __init__(self, title, msg, choices):
        super().__init__(fg_color=ctk.ThemeManager.theme["CTkToplevel"]["fg_color"])
        self.title(title)
        self.lift()
        self.msg = str(msg)
        self.returning = choices[0]
        self.choices = choices
        self.after(10, self.make_ui)
        self.protocol("WM_DELETE_WINDOW", self.close_mod)
        self.resizable(False, False)

    def make_ui(self):
        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)
        message = ctk.CTkLabel(self, text=self.msg)
        message.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        self.c = ctk.CTkComboBox(
            self,
            width=230,
            values=self.choices if self.choices else [],
            state="readonly",
            command=self.combobox_select,
        )
        self.c.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")
        self.c.set(self.choices[0])

        btn_1 = ctk.CTkButton(self, width=8, text="Ok", command=self.b1_action)
        self.bind("<Return>", self.b1_action)
        btn_1.focus_set()
        btn_1.grid(
            row=2, column=0, columnspan=1, padx=(20, 10), pady=(0, 20), sticky="ew"
        )

        self._cancel_button = ctk.CTkButton(
            master=self,
            width=100,
            border_width=0,
            fg_color=("#D30000", "#8B0000"),
            hover_color=("#BF0000", "#610000"),
            text="Cancel",
            command=self.close_mod,
        )
        self._cancel_button.grid(
            row=2, column=1, columnspan=1, padx=(10, 20), pady=(0, 20), sticky="ew"
        )

        btn_1.bind("<KeyPress-Return>", command=self.b1_action)
        # root.update_idletasks()
        # xp = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        # yp = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        # geom = (root.winfo_width(), root.winfo_height(), xp, yp)
        # root.geometry("{}x{}+{}+{}".format(*geom))
        xp = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        yp = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        geom = (xp, yp)
        self.geometry("{}+{}".format(*geom))

    def combobox_select(self, event):
        self.returning = self.c.get()
        print(self.returning)

    def close_mod(self):
        self.destroy()
        os._exit(0)

    def b1_action(self, event=None):
        self.destroy()

    def get_input(self):
        self.master.wait_window(self)
        return self.returning
