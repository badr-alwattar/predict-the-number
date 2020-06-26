import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pyfiglet
#This is to ignore warning messages, you can remove this block from here
import warnings
warnings.filterwarnings("ignore")
#To here

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
# print('\n')
# print('Wlcome to the numbers prediction using machine learning')
# print('To exit the program enter -1')

ascii_banner = pyfiglet.figlet_format("Guess  Me")
print(ascii_banner)
print('coded by: Badr Alwattar')
print('2020 - June')
print('github: badr-alwattar')

print('\n')
print('How to play?')
print('Enter a number between 0 and 360 and guss the number in the image (they are between 0 and 9')
print('Enter your guess and wait for the computer to guess, then you will see the result')
print('Enjoy')
while 1:
	try:		
		print('\n')
		index = int(input("Choose a sample, Enter a number from 0 to 360 (-1 to exit): "))
		if index < 0:
			break
		plt.gray()
		plt.close()
		plt.matshow(digits.images[index])
		plt.xlabel('What am I', fontsize=12)
		plt.show()
		user_predict = int(input("what do you think it is? "))
		if user_predict == digits.target[index]:
			print(pyfiglet.figlet_format("You  Win!!"))
		elif model.predict([digits.data[index]]) == digits.target[index]:
			print(pyfiglet.figlet_format("You  Lose!!"))
		
	except ValueError:
		print('please enter only integer values :)')

print('End of code')


