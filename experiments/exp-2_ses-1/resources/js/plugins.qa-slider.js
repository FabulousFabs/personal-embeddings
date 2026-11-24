// String.format prototype
// as per fearphage on https://stackoverflow.com/a/4673436 (last accessed 28/12/2022 11:59)
if (!String.prototype.format) {
    String.prototype.format = function() {
      var args = arguments;
      return this.replace(/{(\d+)}/g, function(match, number) { 
        return typeof args[number] != 'undefined'
          ? args[number]
          : match
        ;
      });
    };
}

// shake animation
function shakeWord($element) {
    $element.addClass('shake');

    // Remove the class after animation ends
    $element.one('animationend webkitAnimationEnd oAnimationEnd MSAnimationEnd', function () {
        $(this).removeClass('shake');
    });
}

// word categorisation in a circle of images plugin
jsPsych.plugins['qa_slider'] = (function(){
    // language specs
    const lang = {
        'en': {
            'continue': 'Continue',
            'error_no_touch': 'Please give a response to all questions before submitting (all sliders must be touched at least once).'
        },
        'de': {
            'continue': 'Weiter',
            'error_no_touch': 'Achtung! Nicht alle Fragen wurden beantwortet (alle Slider müssen mindestens einmal berühert werden).'
        }
    }

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: 'qa_slider',
        parameters: {
            question: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'How strongly do you agree with the following statements?',
                pretty_name: 'Question',
                description: 'Question to pose for this set of items.'
            },
            items: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Items',
                description: 'Items to present.'
            },
            labels: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: null,
                pretty_name: 'Labels',
                description: 'Labels to present on slider.'
            },
            min: {
                type: jsPsych.plugins.parameterType.INT,
                default: -1,
                pretty_name: 'Minimum value',
                description: 'Minimum value for sliders.'
            },
            max: {
                type: jsPsych.plugins.parameterType.INT,
                default: 1,
                pretty_name: 'Maximum value',
                description: 'Maximum value for sliders.'
            },
            step: {
                type: jsPsych.plugins.parameterType.INT,
                default: "any",
                pretty_name: 'Step size',
                description: 'Step size for sliders (defaults to any).'
            },
            study_language: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'en',
                pretty_name: 'Experiment language',
                description: 'Experiment language.'
            }
        }
    }

    // trial logic
    plugin.trial = function(e, t) {
        // log start of trial
        console.log("Trial", t, ": Starting...");

        // setup data structure
        let data = {
            question: t.question
        };

        // clear canvas
        e.innerHTML = '';

        // setup task container
        let taskContainer = $(`
            <div id="qa-task" class="qa-task">
                <div id="qa-header" class="qa-header">
                <div id="qa-question" class="qa-question">${t.question}</div>
                </div>
                <div id="qa-answers" class="qa-answers"></div><br />
                <div id="qa-footer">
                    <button id="qa-submit" class="jspsych-btn">${lang[t.study_language]['continue']}</button>
                </div>
            </div>
        `);

        taskContainer.appendTo(e);
        let answerContainer = taskContainer.find("#qa-answers");
        let buttonSubmit = taskContainer.find('#qa-submit');

        // fallback scale labels
        const scaleLabels = t.labels || [
            { value: t.min, label: "low" },
            { value: (t.min + t.max) / 2, label: "medium" },
            { value: t.max, label: "high" }
        ];

        t.items.forEach((v, i) => {
            const sliderId = `qa-item-${i}-inp`;

            const labelHtml = scaleLabels.map(s => {
                const pct = ((s.value - t.min) / (t.max - t.min)) * 100;
                return `
                <span class="qa-scale-label" style="left:${pct}%;">
                    ${s.label}
                </span>
                `;
            }).join("");

            const item = $(`
                <div id="qa-item-${i}" class="qa-item">
                    <label for="${sliderId}" class="qa-item-label">${v}</label>
                    <div class="qa-slider-wrap">
                        <input
                            id="${sliderId}"
                            name="${sliderId}"
                            class="qa-slider"
                            previously-touched="false"
                            type="range"
                            min="${t.min}"
                            max="${t.max}"
                            step="${t.step}"
                            value="${t.slider_start ?? ((t.min + t.max) / 2)}"
                        />
                        <div class="qa-scale">
                            ${labelHtml}
                        </div>
                    </div>
                </div>
            `);

            item.appendTo(answerContainer);
        });

        // setup touched tracking
        $('.qa-slider').on('mousedown', function(event){
            $(this).attr('previously-touched', 'true');
            $(this).removeClass('qa-error');
        });

        // setup submission
        buttonSubmit.on('mouseup', function(event){
            // check item status
            let all_touched = true;

            $('.qa-slider').each(function(){
                if ($(this).attr('previously-touched') != 'true'){
                    // disable submission
                    all_touched = false;

                    // alert user by shake
                    shakeWord($(this));

                    // highlight item
                    $(this).addClass('qa-error');
                }
            });

            if (all_touched) {
                // collect data for submission
                $('.qa-item').each(function(count){
                    let item = $(this).find('.qa-item-label').text();
                    let value = $(this).find('input').val();
                    data[`item_${count}`] = item;
                    data[`value_${count}`] = value;
                });

                console.log(data);

                // make submission
                jsPsych.pluginAPI.clearAllTimeouts();
                jsPsych.finishTrial(data);
            } else {
                // not all items have been touched
                alert(lang[t.study_language]['error_no_touch']);
            }
        });
    }

    return plugin;
})();