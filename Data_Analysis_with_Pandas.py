import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import random

# data = [10, 20, 30, 40, 50]
# series = pd.Series(data)
# print(series)

# class NumberGuessingGame:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Guess the Number Game")
#         self.root.geometry("400x300")
#         self.root.resizable(False, False)
        
#         self.secret_number = random.randint(1, 10)
#         self.attempts = 0
#         self.max_attempts = 5
        
#         self.create_widgets()
        
#     def create_widgets(self):
#         bg_color = "#e2d1d1"
#         button_color = "#2486aa"
#         text_color = "#3F3838"
        
#         self.root.configure(bg=bg_color)
        
#         title = tk.Label(
#             self.root, 
#             text="Guess the Number (1-10)",
#             font=("Arial", 16, "bold"),
#             bg=bg_color,
#             fg=text_color
#         )
#         title.pack(pady=10)
        
#         self.attempts_label = tk.Label(
#             self.root,
#             text=f"Attempts: {self.attempts}/{self.max_attempts}",
#             font=("Arial", 10),
#             bg=bg_color,
#             fg=text_color
#         )
#         self.attempts_label.pack()
        
#         self.hint_label = tk.Label(
#             self.root,
#             text="I'm thinking of a number between 1 and 10",
#             font=("Arial", 10, "italic"),
#             bg=bg_color,
#             fg=text_color
#         )
#         self.hint_label.pack(pady=5)
        
#         self.guess_entry = tk.Entry(
#             self.root,
#             font=("Arial", 12),
#             width=10,
#             justify="center"
#         )
#         self.guess_entry.pack(pady=10)
#         self.guess_entry.focus()
        
#         submit_btn = tk.Button(
#             self.root,
#             text="Submit Guess",
#             command=self.check_guess,
#             bg=button_color,
#             fg="white",
#             font=("Arial", 10, "bold"),
#             padx=10,
#             pady=5
#         )
#         submit_btn.pack(pady=5)
        
#         self.root.bind('<Return>', lambda event: self.check_guess())
        
#     def check_guess(self):
#         guess = self.guess_entry.get()
        
#         try:
#             guess = int(guess)
#             if guess < 1 or guess > 100:
#                 messagebox.showerror("Error", "Please enter a number between 1 and 10!")
#                 return
                
#             self.attempts += 1
#             self.attempts_label.config(text=f"Attempts: {self.attempts}/{self.max_attempts}")
            
#             if guess == self.secret_number:
#                 messagebox.showinfo(
#                     "Congratulations!",
#                     f"You guessed the number {self.secret_number} in {self.attempts} attempts!"
#                 )
#                 self.play_again()
#             elif self.attempts >= self.max_attempts:
#                 messagebox.showinfo(
#                     "Game Over",
#                     f"Sorry! The number was {self.secret_number}. Better luck next time!"
#                 )
#                 self.play_again()
#             else:
#                 if guess < self.secret_number:
#                     self.hint_label.config(text="The number is higher!")
#                 else:
#                     self.hint_label.config(text="The number is lower!")
                
#         except ValueError:
#             messagebox.showerror("Error", "Please enter a valid number!")
        
#         self.guess_entry.delete(0, tk.END)
    
#     def play_again(self):
#         answer = messagebox.askyesno(
#             "Play Again?",
#             "Would you like to play again?"
#         )
        
#         if answer:
#             self.secret_number = random.randint(1, 10)
#             self.attempts = 0
#             self.attempts_label.config(text=f"Attempts: {self.attempts}/{self.max_attempts}")
#             self.hint_label.config(text="I'm thinking of a number between 1 and 10")
#         else:
#             self.root.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     game = NumberGuessingGame(root)
#     root.mainloop()




# data = {
#     'Date': pd.date_range(start='2023-01-01', periods=30),
#     'Product': ['A', 'B', 'C']*10,
#     'Quantity': [5, 3, 8, 6, 2, 7, 4, 5, 6, 3]*3,
#     'Unit_Price': [100, 150, 80, 120, 200, 90, 110, 100, 130, 140]*3
# }

# df = pd.DataFrame(data)
# df['Total_Sales'] = df['Quantity'] * df['Unit_Price']

# print("=== Sales Summary ===")
# print(f"Total Sales: ${df['Total_Sales'].sum():,.2f}")
# print(f"Average Daily Sales: ${df['Total_Sales'].mean():,.2f}")
# print("\nTop Selling Products:")
# print(df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False))

# plt.figure(figsize=(10,4))
# df.groupby('Date')['Total_Sales'].sum().plot(kind='line', title='Daily Sales Trend')
# plt.ylabel('Sales ($)')
# plt.grid(True)
# plt.show()




customer_data = {
    'CustomerID': [101, 102, 103, 104, 105],
    'Last_Purchase': ['2023-05-10', '2023-06-15', '2023-01-20', '2023-06-01', '2023-04-05'],
    'Purchase_Count': [5, 12, 3, 8, 6],
    'Total_Spent': [1200, 4500, 800, 3600, 2100]
}

customers = pd.DataFrame(customer_data)
customers['Last_Purchase'] = pd.to_datetime(customers['Last_Purchase'])

current_date = pd.to_datetime('2023-06-20')
rfm = customers.copy()
rfm['Recency'] = (current_date - rfm['Last_Purchase']).dt.days
rfm['Frequency'] = rfm['Purchase_Count']
rfm['Monetary'] = rfm['Total_Spent']

rfm['R_Score'] = pd.qcut(rfm['Recency'], q=3, labels=[3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=3, labels=[1,2,3])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=3, labels=[1,2,3])

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print("\n=== Customer Segmentation ===")
print(rfm[['CustomerID', 'RFM_Score']].sort_values('RFM_Score', ascending=False))

plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Frequency'], cmap='viridis')
plt.xlabel('Days Since Last Purchase')
plt.ylabel('Total Amount Spent ($)')
plt.title('Customer Value Analysis')
plt.colorbar(label='Purchase Frequency')
plt.show()