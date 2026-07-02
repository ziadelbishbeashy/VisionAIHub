def answer_weather_question(question, predicted_weather):
    question = (question or "").strip().lower()
    weather = (predicted_weather or "").strip().title()

    if not question:
        return "Please enter a weather-related question."

    if "umbrella" in question:
        if weather == "Rainy":
            return "Yes, carrying an umbrella is recommended."
        return "No, an umbrella is probably not needed."

    if "sunglasses" in question or "sun glasses" in question:
        if weather == "Shine":
            return "Yes, sunglasses are recommended."
        return "Sunglasses are not necessary."

    if "jacket" in question:
        if weather == "Rainy":
            return "Yes, wearing a waterproof jacket is recommended."

        if weather == "Cloudy":
            return "A light jacket may be useful."

        return "A jacket is probably not needed."

    if "weather" in question:
        return f"The predicted weather is {weather}."

    return (
        "Sorry, I currently answer questions about umbrellas, "
        "sunglasses, jackets, and the predicted weather."
    )