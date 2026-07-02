document.addEventListener(
    "DOMContentLoaded",
    () => {
        initializeAnalysisSelector();
        initializeVisionAssistant();
    }
);


/* =========================================================
   ANALYSIS MODULE SELECTION
========================================================= */

function initializeAnalysisSelector() {
    const analysisForm =
        document.getElementById(
            "analysis-form"
        );

    if (!analysisForm) {
        return;
    }


    const checkboxes = Array.from(
        document.querySelectorAll(
            'input[name="analysis_types"]'
        )
    );


    const options = Array.from(
        document.querySelectorAll(
            ".analysis-option"
        )
    );


    const presetButtons = Array.from(
        document.querySelectorAll(
            ".preset-button"
        )
    );


    const detectionCheckbox =
        document.querySelector(
            'input[value="detection"]'
        );


    const trafficCheckbox =
        document.querySelector(
            'input[value="traffic"]'
        );


    const weatherCheckbox =
        document.querySelector(
            'input[value="weather"]'
        );


    const sceneCheckbox =
        document.querySelector(
            'input[value="scene"]'
        );


    const confidenceSection =
        document.getElementById(
            "confidence-section"
        );


    const moduleError =
        document.getElementById(
            "module-error"
        );


    const imageInput =
        document.getElementById(
            "image-input"
        );


    const fileName =
        document.getElementById(
            "file-name"
        );


    const submitButton =
        analysisForm.querySelector(
            ".run-analysis-button"
        );


    /* =====================================================
       LOADING SCREEN ELEMENTS
    ===================================================== */

    const loadingOverlay =
        document.getElementById(
            "loading-overlay"
        );


    const loadingStage =
        document.getElementById(
            "loading-stage"
        );


    const loadingModule =
        document.getElementById(
            "loading-module"
        );


    const loadingPercentage =
        document.getElementById(
            "loading-percentage"
        );


    const loadingProgressBar =
        document.getElementById(
            "loading-progress-bar"
        );


    const loadingYolo =
        document.getElementById(
            "loading-yolo"
        );


    const loadingTraffic =
        document.getElementById(
            "loading-traffic"
        );


    const loadingWeather =
        document.getElementById(
            "loading-weather"
        );


    const loadingScene =
        document.getElementById(
            "loading-scene"
        );


    /* =====================================================
       UPDATE MODULE CARD APPEARANCE
    ===================================================== */

    function updateOptionStyles() {
        options.forEach((option) => {
            const checkbox =
                option.querySelector(
                    'input[type="checkbox"]'
                );

            if (!checkbox) {
                return;
            }

            option.classList.toggle(
                "selected",
                checkbox.checked
            );
        });


        const needsDetection =
            Boolean(
                detectionCheckbox?.checked
                || trafficCheckbox?.checked
            );


        if (confidenceSection) {
            confidenceSection.style.display =
                needsDetection
                    ? "block"
                    : "none";
        }
    }


    /* =====================================================
       TRAFFIC DEPENDENCY
    ===================================================== */

    function enforceTrafficDependency() {
        if (
            trafficCheckbox
            && trafficCheckbox.checked
            && detectionCheckbox
        ) {
            detectionCheckbox.checked = true;
        }

        updateOptionStyles();
    }


    /* =====================================================
       CHECKBOX EVENTS
    ===================================================== */

    checkboxes.forEach((checkbox) => {
        checkbox.addEventListener(
            "change",
            () => {
                enforceTrafficDependency();

                if (moduleError) {
                    moduleError.textContent = "";
                }

                presetButtons.forEach(
                    (button) => {
                        button.classList.remove(
                            "active"
                        );
                    }
                );
            }
        );
    });


    /* =====================================================
       PRESET BUTTONS
    ===================================================== */

    presetButtons.forEach((button) => {
        button.addEventListener(
            "click",
            () => {
                const preset =
                    button.dataset.preset;


                if (detectionCheckbox) {
                    detectionCheckbox.checked = false;
                }

                if (trafficCheckbox) {
                    trafficCheckbox.checked = false;
                }

                if (weatherCheckbox) {
                    weatherCheckbox.checked = false;
                }

                if (sceneCheckbox) {
                    sceneCheckbox.checked = false;
                }


                if (preset === "all") {
                    if (detectionCheckbox) {
                        detectionCheckbox.checked = true;
                    }

                    if (trafficCheckbox) {
                        trafficCheckbox.checked = true;
                    }

                    if (weatherCheckbox) {
                        weatherCheckbox.checked = true;
                    }

                    if (sceneCheckbox) {
                        sceneCheckbox.checked = true;
                    }
                }


                if (preset === "traffic") {
                    if (detectionCheckbox) {
                        detectionCheckbox.checked = true;
                    }

                    if (trafficCheckbox) {
                        trafficCheckbox.checked = true;
                    }
                }


                if (preset === "weather") {
                    if (weatherCheckbox) {
                        weatherCheckbox.checked = true;
                    }
                }


                if (preset === "scene") {
                    if (sceneCheckbox) {
                        sceneCheckbox.checked = true;
                    }
                }


                if (preset === "detection") {
                    if (detectionCheckbox) {
                        detectionCheckbox.checked = true;
                    }
                }


                presetButtons.forEach(
                    (presetButton) => {
                        presetButton.classList.remove(
                            "active"
                        );
                    }
                );


                button.classList.add(
                    "active"
                );


                enforceTrafficDependency();


                if (moduleError) {
                    moduleError.textContent = "";
                }
            }
        );
    });


    /* =====================================================
       IMAGE FILE NAME
    ===================================================== */

    if (imageInput) {
        imageInput.addEventListener(
            "change",
            () => {
                if (
                    imageInput.files
                    && imageInput.files.length > 0
                ) {
                    fileName.textContent =
                        imageInput.files[0].name;

                } else {
                    fileName.textContent =
                        "Choose an image";
                }


                if (moduleError) {
                    moduleError.textContent = "";
                }
            }
        );
    }


    /* =====================================================
       FORM SUBMISSION
    ===================================================== */

    analysisForm.addEventListener(
        "submit",
        (event) => {
            enforceTrafficDependency();


            const selectedCount =
                checkboxes.filter(
                    (checkbox) =>
                        checkbox.checked
                ).length;


            if (selectedCount === 0) {
                event.preventDefault();

                if (moduleError) {
                    moduleError.textContent =
                        "Select at least one analysis module.";
                }

                return;
            }


            if (
                !imageInput
                || !imageInput.files
                || imageInput.files.length === 0
            ) {
                event.preventDefault();

                if (moduleError) {
                    moduleError.textContent =
                        "Please choose an image.";
                }

                return;
            }


            if (moduleError) {
                moduleError.textContent = "";
            }


            showLoadingScreen();
        }
    );


    /* =====================================================
       SHOW LOADING SCREEN
    ===================================================== */

    function showLoadingScreen() {
        if (
            !loadingOverlay
            || !loadingStage
            || !loadingModule
            || !loadingPercentage
            || !loadingProgressBar
        ) {
            if (submitButton) {
                submitButton.disabled = true;

                submitButton.textContent =
                    "Analysis Running...";
            }

            return;
        }


        const stages = [];


        const detectionSelected =
            Boolean(
                detectionCheckbox?.checked
                || trafficCheckbox?.checked
            );


        const trafficSelected =
            Boolean(
                trafficCheckbox?.checked
            );


        const weatherSelected =
            Boolean(
                weatherCheckbox?.checked
            );


        const sceneSelected =
            Boolean(
                sceneCheckbox?.checked
            );


        updateLoadingModelStatus(
            loadingYolo,
            detectionSelected
        );


        updateLoadingModelStatus(
            loadingTraffic,
            trafficSelected
        );


        updateLoadingModelStatus(
            loadingWeather,
            weatherSelected
        );


        updateLoadingModelStatus(
            loadingScene,
            sceneSelected
        );


        stages.push({
            text:
                "Uploading and validating the image...",

            module:
                "Image preparation",

            progress: 8
        });


        if (detectionSelected) {
            stages.push({
                text:
                    "Detecting people, vehicles and road objects...",

                module:
                    "YOLOv8 object detection",

                progress: 25
            });
        }


        if (trafficSelected) {
            stages.push({
                text:
                    "Creating occupancy, density and traffic features...",

                module:
                    "Feature engineering · 25 features",

                progress: 43
            });


            stages.push({
                text:
                    "Classifying traffic density...",

                module:
                    "Random Forest prediction",

                progress: 58
            });
        }


        if (weatherSelected) {
            stages.push({
                text:
                    "Analyzing the weather condition...",

                module:
                    "Weather transfer-learning model",

                progress: 72
            });
        }


        if (sceneSelected) {
            stages.push({
                text:
                    "Classifying the complete scene...",

                module:
                    "Scene transfer-learning model",

                progress: 84
            });
        }


        stages.push({
            text:
                "Combining model predictions...",

            module:
                "Multi-model analysis",

            progress: 92
        });


        stages.push({
            text:
                "Preparing your results dashboard...",

            module:
                "Finalizing results",

            progress: 97
        });


        loadingOverlay.classList.add(
            "active"
        );


        loadingOverlay.setAttribute(
            "aria-hidden",
            "false"
        );


        document.body.classList.add(
            "loading-active"
        );


        if (submitButton) {
            submitButton.disabled = true;

            submitButton.textContent =
                "Analysis Running...";
        }


        let currentStageIndex = 0;


        displayLoadingStage(
            stages[currentStageIndex]
        );


        const loadingInterval =
            window.setInterval(
                () => {
                    if (
                        currentStageIndex
                        < stages.length - 1
                    ) {
                        currentStageIndex += 1;

                        displayLoadingStage(
                            stages[currentStageIndex]
                        );

                        return;
                    }


                    loadingStage.textContent =
                        "Finalizing the analysis. Please wait...";


                    loadingModule.textContent =
                        "Building results page";


                    const currentProgress =
                        Number(
                            loadingProgressBar
                                .dataset
                                .progress
                            || 97
                        );


                    const updatedProgress =
                        Math.min(
                            currentProgress + 1,
                            99
                        );


                    setLoadingProgress(
                        updatedProgress
                    );

                },
                1100
            );


        loadingOverlay.dataset.intervalId =
            String(loadingInterval);
    }


    /* =====================================================
       DISPLAY LOADING STAGE
    ===================================================== */

    function displayLoadingStage(stage) {
        if (!stage) {
            return;
        }


        loadingStage.textContent =
            stage.text;


        loadingModule.textContent =
            stage.module;


        setLoadingProgress(
            stage.progress
        );
    }


    /* =====================================================
       UPDATE LOADING PROGRESS
    ===================================================== */

    function setLoadingProgress(progress) {
        const safeProgress =
            Math.min(
                Math.max(
                    Number(progress),
                    0
                ),
                100
            );


        loadingProgressBar.style.width =
            `${safeProgress}%`;


        loadingProgressBar.dataset.progress =
            String(safeProgress);


        loadingPercentage.textContent =
            `${safeProgress}%`;
    }


    /* =====================================================
       UPDATE MODEL BADGES
    ===================================================== */

    function updateLoadingModelStatus(
        element,
        isSelected
    ) {
        if (!element) {
            return;
        }


        element.classList.toggle(
            "active",
            isSelected
        );


        element.classList.toggle(
            "inactive",
            !isSelected
        );
    }


    updateOptionStyles();
}


/* =========================================================
   GEMINI CONTEXT-AWARE ASSISTANT
========================================================= */

function initializeVisionAssistant() {
    const form =
        document.getElementById(
            "analysis-question-form"
        );


    const input =
        document.getElementById(
            "analysis-question"
        );


    const sendButton =
        document.getElementById(
            "analysis-send-button"
        );


    const chat =
        document.getElementById(
            "analysis-chat"
        );


    const errorText =
        document.getElementById(
            "analysis-assistant-error"
        );


    const statusText =
        document.getElementById(
            "analysis-assistant-status"
        );


    const suggestionButtons =
        document.querySelectorAll(
            ".analysis-suggestion"
        );


    if (
        !form
        || !input
        || !sendButton
        || !chat
    ) {
        return;
    }


    /* =====================================================
       ADD CHAT MESSAGE
    ===================================================== */

    function addMessage(
        text,
        sender
    ) {
        const message =
            document.createElement(
                "div"
            );


        message.className =
            sender === "user"
                ? "chat-message user-message"
                : "chat-message assistant-message";


        const label =
            document.createElement(
                "div"
            );


        label.className =
            "message-label";


        label.textContent =
            sender === "user"
                ? "You"
                : "Gemini Assistant";


        const paragraph =
            document.createElement(
                "p"
            );


        paragraph.textContent =
            text;


        message.appendChild(
            label
        );


        message.appendChild(
            paragraph
        );


        chat.appendChild(
            message
        );


        chat.scrollTop =
            chat.scrollHeight;
    }


    /* =====================================================
       LOADING STATE
    ===================================================== */

    function setAssistantLoading(
        isLoading
    ) {
        input.disabled =
            isLoading;


        sendButton.disabled =
            isLoading;


        sendButton.textContent =
            isLoading
                ? "Thinking..."
                : "Ask Gemini";


        if (
            isLoading
            && statusText
        ) {
            statusText.textContent =
                "Thinking...";
        }
    }


    /* =====================================================
       SEND QUESTION TO FLASK
    ===================================================== */

    async function askAssistant(
        question
    ) {
        const cleanQuestion =
            String(
                question || ""
            ).trim();


        if (!cleanQuestion) {
            if (errorText) {
                errorText.textContent =
                    "Please enter a question.";
            }

            return;
        }


        if (errorText) {
            errorText.textContent = "";
        }


        addMessage(
            cleanQuestion,
            "user"
        );


        input.value = "";


        setAssistantLoading(
            true
        );


        try {
            const response =
                await fetch(
                    "/ask-assistant",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            question:
                                cleanQuestion
                        })
                    }
                );


            let data;


            try {
                data =
                    await response.json();

            } catch {
                throw new Error(
                    "The assistant returned an invalid response."
                );
            }


            if (
                !response.ok
                || !data.success
            ) {
                throw new Error(
                    data.answer
                    || data.error
                    || "The assistant could not answer."
                );
            }


            addMessage(
                data.answer,
                "assistant"
            );


            if (statusText) {
                statusText.textContent =
                    data.assistant_type
                    || "Gemini";
            }

        } catch (error) {
            console.error(
                "Gemini assistant error:",
                error
            );


            if (errorText) {
                errorText.textContent =
                    error.message;
            }


            addMessage(
                (
                    "The AI assistant is temporarily "
                    + "unavailable. Please try again."
                ),
                "assistant"
            );


            if (statusText) {
                statusText.textContent =
                    "Unavailable";
            }

        } finally {
            setAssistantLoading(
                false
            );


            input.focus();
        }
    }


    /* =====================================================
       FORM SUBMISSION
    ===================================================== */

    form.addEventListener(
        "submit",
        (event) => {
            event.preventDefault();


            askAssistant(
                input.value
            );
        }
    );


    /* =====================================================
       SUGGESTED QUESTIONS
    ===================================================== */

    suggestionButtons.forEach(
        (button) => {
            button.addEventListener(
                "click",
                () => {
                    askAssistant(
                        button.dataset.question
                    );
                }
            );
        }
    );
}