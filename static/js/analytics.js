document.addEventListener(
    "DOMContentLoaded",
    () => {
        const analyticsDataElement =
            document.getElementById(
                "analytics-data"
            );

        let data = {};

        if (analyticsDataElement) {
            try {
                data = JSON.parse(
                    analyticsDataElement.textContent
                );
            } catch (error) {
                console.error(
                    "Could not read analytics data:",
                    error
                );
            }
        }


        if (typeof Chart === "undefined") {
            console.error(
                "Chart.js was not loaded."
            );

            return;
        }


        createLineChart(
            "analyses-over-time-chart",
            data.analyses_over_time || {},
            "Analyses"
        );


        createBarChart(
            "object-distribution-chart",
            data.object_distribution || {},
            "Detected Objects",
            true
        );


        createDoughnutChart(
            "traffic-distribution-chart",
            data.traffic_distribution || {}
        );


        createDoughnutChart(
            "weather-distribution-chart",
            data.weather_distribution || {}
        );


        createBarChart(
            "scene-distribution-chart",
            data.scene_distribution || {},
            "Predictions",
            true
        );


        createBarChart(
            "confidence-comparison-chart",
            data.average_confidence || {},
            "Average Confidence (%)",
            false,
            100
        );


        createDoughnutChart(
            "people-vehicles-chart",
            data.people_vehicles || {}
        );
    }
);


/* =========================================================
   CLEAN AND VALIDATE DATA
========================================================= */

function validEntries(dataObject) {
    return Object.entries(
        dataObject || {}
    ).filter(
        ([, value]) =>
            Number.isFinite(
                Number(value)
            )
    );
}


/* =========================================================
   LINE CHART
========================================================= */

function createLineChart(
    canvasId,
    dataObject,
    datasetLabel
) {
    const canvas =
        document.getElementById(
            canvasId
        );


    if (!canvas) {
        return;
    }


    const entries =
        validEntries(
            dataObject
        );


    if (entries.length === 0) {
        showEmptyChartMessage(
            canvas,
            "No data available"
        );

        return;
    }


    new Chart(
        canvas,
        {
            type: "line",

            data: {
                labels:
                    entries.map(
                        ([label]) =>
                            label
                    ),

                datasets: [
                    {
                        label:
                            datasetLabel,

                        data:
                            entries.map(
                                ([, value]) =>
                                    Number(value)
                            ),

                        borderWidth: 3,
                        tension: 0.35,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                interaction: {
                    mode: "index",
                    intersect: false
                },

                plugins: {
                    legend: {
                        display: false
                    },

                    tooltip: {
                        callbacks: {
                            label(context) {
                                return (
                                    `${datasetLabel}: `
                                    + `${context.raw}`
                                );
                            }
                        }
                    }
                },

                scales: {
                    x: {
                        grid: {
                            display: false
                        },

                        title: {
                            display: true,
                            text: "Date"
                        }
                    },

                    y: {
                        beginAtZero: true,

                        ticks: {
                            precision: 0
                        },

                        title: {
                            display: true,
                            text: "Number of Analyses"
                        }
                    }
                }
            }
        }
    );
}


/* =========================================================
   BAR CHART
========================================================= */

function createBarChart(
    canvasId,
    dataObject,
    datasetLabel,
    horizontal = false,
    maximum = null
) {
    const canvas =
        document.getElementById(
            canvasId
        );


    if (!canvas) {
        return;
    }


    const entries =
        validEntries(
            dataObject
        );


    if (entries.length === 0) {
        showEmptyChartMessage(
            canvas,
            "No data available"
        );

        return;
    }


    const values =
        entries.map(
            ([, value]) =>
                Number(value)
        );


    const scales = horizontal
        ? {
            x: {
                beginAtZero: true,
                max:
                    maximum !== null
                        ? maximum
                        : undefined,

                ticks: {
                    precision: 0
                },

                title: {
                    display: true,
                    text:
                        maximum === 100
                            ? "Confidence (%)"
                            : "Count"
                }
            },

            y: {
                grid: {
                    display: false
                }
            }
        }

        : {
            x: {
                grid: {
                    display: false
                }
            },

            y: {
                beginAtZero: true,
                max:
                    maximum !== null
                        ? maximum
                        : undefined,

                ticks: {
                    precision: 0
                },

                title: {
                    display: true,
                    text:
                        maximum === 100
                            ? "Confidence (%)"
                            : "Count"
                }
            }
        };


    new Chart(
        canvas,
        {
            type: "bar",

            data: {
                labels:
                    entries.map(
                        ([label]) =>
                            label
                    ),

                datasets: [
                    {
                        label:
                            datasetLabel,

                        data:
                            values,

                        borderRadius: 8,
                        borderSkipped: false
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                indexAxis:
                    horizontal
                        ? "y"
                        : "x",

                plugins: {
                    legend: {
                        display: false
                    },

                    tooltip: {
                        callbacks: {
                            label(context) {
                                const suffix =
                                    maximum === 100
                                        ? "%"
                                        : "";

                                return (
                                    `${datasetLabel}: `
                                    + `${context.raw}${suffix}`
                                );
                            }
                        }
                    }
                },

                scales:
                    scales
            }
        }
    );
}


/* =========================================================
   DOUGHNUT CHART
========================================================= */

function createDoughnutChart(
    canvasId,
    dataObject
) {
    const canvas =
        document.getElementById(
            canvasId
        );


    if (!canvas) {
        return;
    }


    const entries =
        validEntries(
            dataObject
        );


    const filteredEntries =
        entries.filter(
            ([, value]) =>
                Number(value) > 0
        );


    if (filteredEntries.length === 0) {
        showEmptyChartMessage(
            canvas,
            "No data available"
        );

        return;
    }


    new Chart(
        canvas,
        {
            type: "doughnut",

            data: {
                labels:
                    filteredEntries.map(
                        ([label]) =>
                            label
                    ),

                datasets: [
                    {
                        data:
                            filteredEntries.map(
                                ([, value]) =>
                                    Number(value)
                            ),

                        borderWidth: 0,
                        hoverOffset: 8
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "62%",

                plugins: {
                    legend: {
                        position: "bottom",

                        labels: {
                            usePointStyle: true,
                            padding: 18
                        }
                    },

                    tooltip: {
                        callbacks: {
                            label(context) {
                                const label =
                                    context.label || "";

                                const value =
                                    Number(
                                        context.raw
                                    );

                                const total =
                                    context.dataset.data
                                        .reduce(
                                            (
                                                sum,
                                                item
                                            ) =>
                                                sum
                                                + Number(item),
                                            0
                                        );

                                const percentage =
                                    total > 0
                                        ? (
                                            value
                                            / total
                                            * 100
                                        ).toFixed(1)
                                        : 0;

                                return (
                                    `${label}: `
                                    + `${value} `
                                    + `(${percentage}%)`
                                );
                            }
                        }
                    }
                }
            }
        }
    );
}


/* =========================================================
   EMPTY CHART MESSAGE
========================================================= */

function showEmptyChartMessage(
    canvas,
    message
) {
    const container =
        canvas.parentElement;


    if (!container) {
        return;
    }


    canvas.style.display =
        "none";


    const emptyMessage =
        document.createElement(
            "div"
        );


    emptyMessage.className =
        "chart-empty-message";


    emptyMessage.textContent =
        message;


    container.appendChild(
        emptyMessage
    );
}