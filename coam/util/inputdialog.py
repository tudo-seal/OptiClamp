import os

import customtkinter as ctk


class CTkInputDialog(ctk.CTkToplevel):
    """
    Dialog with extra window, message, entry widget, cancel and ok button.
    For detailed information check out the documentation.
    """

    def __init__(
        self,
        text: str = "CTkDialog",
        title: str = "CTkDialog",
        entry_text: str | None = None,
    ):
        super().__init__(fg_color=ctk.ThemeManager.theme["CTkToplevel"]["fg_color"])
        self.title(title)
        self.lift()
        self._label_text = text
        self._entry_text = entry_text
        self._user_input: str | None = None
        self.after(10, self.make_ui)
        self.protocol("WM_DELETE_WINDOW", self.close_mod)
        self.resizable(False, False)

    def make_ui(self):
        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)
        self._label = ctk.CTkLabel(
            master=self,
            width=300,
            wraplength=300,
            fg_color="transparent",
            text=self._label_text,
        )
        self._label.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        self._entry = ctk.CTkEntry(
            master=self,
            width=230,
            textvariable=ctk.StringVar(self, self._entry_text),
        )
        self._entry.grid(
            row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew"
        )

        self._ok_button = ctk.CTkButton(
            master=self,
            width=100,
            border_width=0,
            text="Ok",
            command=self._ok_event,
        )
        self._ok_button.grid(
            row=2, column=0, columnspan=1, padx=(20, 10), pady=(0, 20), sticky="ew"
        )

        self._cancel_button = ctk.CTkButton(
            master=self,
            width=100,
            border_width=0,
            fg_color=("#D30000", "#8B0000"),
            hover_color=("#BF0000", "#610000"),
            text="Cancel",
            command=self._cancel_event,
        )
        self._cancel_button.grid(
            row=2, column=1, columnspan=1, padx=(10, 20), pady=(0, 20), sticky="ew"
        )

        self.bind("<Return>", self._ok_event)
        self._ok_button.focus_set()
        xp = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        yp = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        geom = (xp, yp)
        self.geometry("{}+{}".format(*geom))

    def _ok_event(self, event=None):
        self._user_input = self._entry.get()
        self.destroy()

    def close_mod(self):
        self.destroy()
        os._exit(0)

    def _on_closing(self):
        self.destroy()

    def _cancel_event(self):
        self.destroy()
        os._exit(0)

    def get_input(self):
        self.master.wait_window(self)
        return self._user_input
