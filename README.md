**Practical 1:**

In the exam, on the college computer:

Open the Terminal with Ctrl + Alt + T.
Make a folder and go into it:
   mkdir ml_exam
   cd ml_exam
Copy uber.csv into this folder. You can drag it in with the file manager, or run cp /path/to/uber.csv ..
Check the libraries are installed:
   python3 -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('OK')"

If it prints OK, skip to step 6.
5. If a library is missing, install it:

   pip3 install pandas numpy matplotlib seaborn scikit-learn

If pip3 is not found, run sudo apt install python3-pip first. If you get an “externally-managed-environment” error, add --break-system-packages at the end of the pip command.
6. Create the code file with nano uber.py, paste the code, then press Ctrl+O, Enter, Ctrl+X to save and exit. You can also type it in the Text Editor and save it as uber.py in the same folder.
7. Run it:

   python3 uber.py
Close each plot window to move to the next section. The scores print in the terminal at the end.

Common problems:

FileNotFoundError: uber.csv means you are in the wrong folder. Run ls to check that uber.csv is listed.
If the random forest is slow, change n_estimators=50 to 20.
If plots don’t open, replace plt.show() with plt.savefig('plot1.png') (use a different name each time).
