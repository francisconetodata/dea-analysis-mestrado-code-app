
import datetime

import telebot

from .secrets import api_telegram


def alguemfezlogin():
    bot = telebot.TeleBot(
        api_telegram
    )

    bot.send_message('980742303', 'Alguém acabou de login às: ' +
                     str(datetime.datetime.now() - datetime.timedelta(hours=3)))

    # bot.polling()


def alguemfezanalise():
    bot = telebot.TeleBot(
        api_telegram
    )

    bot.send_message('980742303', 'Alguém acabou de fazer análise às: ' +
                     str(datetime.datetime.now() - datetime.timedelta(hours=3)))

    # bot.polling()


def alguemfezdownload():
    bot = telebot.TeleBot(
        api_telegram
    )

    bot.send_message('980742303', 'Alguém acabou de fazer download dados às: ' +
                     str(datetime.datetime.now() - datetime.timedelta(hours=3)))

    # bot.polling()


def alguemfezanalisedea():
    bot = telebot.TeleBot(
        api_telegram
    )

    bot.send_message('980742303', 'Alguém acabou de fazer análise DEA às: ' +
                     str(datetime.datetime.now() - datetime.timedelta(hours=3)))

    # bot.polling()
