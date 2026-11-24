// helper functions for instructions & briefing
function paragraphs(...strings) {
    return strings.map(s => {
        return '<p>' + s + '</p>';
    }).join("");
}

function bold(string) {
    return '<b>' + string + '</b>'
}

function italic(string) {
    return '<i>' + string + '</i>'
}

function ulist(...strings) {
    return '<ul>' + strings.map(s => {
        return '<li>' + s + '</li>'
    }).join("")  + '</ul>'
}

function cell(string) {
    return '<td>' + string + '</td>'
}

function cells(...strings) {
    return strings.map(s => {
        return cell(s)
    }).join("");
}

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

// main plugin
jsPsych.plugins['informed_consent_uke'] = (function(){
    // language settings
    const lang = {
        'de': {
            // navigation text
            'navigation_previous': 'Zurück',
            'navigation_continue': 'Weiter',

            // text for general briefing
            'general_header': 'Informationen, Aufklärung, Einwilligung',
            'general_greeting': 'Sehr geehrte Probandin, sehr geehrter Proband,',
            'general_introduction': 'im Rahmen eines aktuellen Forschungsprojektes untersucht {0}{1}.',
            'general_procedure': 'Ablauf',
            'general_inclusion': 'Bitte nehmen Sie nur teil, wenn Sie die folgenden Einschlusskriterien erfüllen:',
            'general_exclusion': 'Bitte nehmen Sie NICHT teil, wenn sie eines der folgenden Ausschlusskriterien erfüllen:',
            'general_signoff': 'Wir würden uns über Ihre Teilnahme sehr freuen und bedanken uns bereits jetzt herzlich bei Ihnen.',
            'general_and': 'und',

            // text for privacy briefing
            'privacy_header': 'Informationen zur Verarbeitung Ihrer personenbezogenen Daten',
            'privacy_storage_header': '1. Speicherung der erhobenen Daten',
            'privacy_storage_declaration': 'Dieses Online Experiment wird über Pavlovia durchgeführt. Pavlovia ist konform mit der EU General Data Protection Regulation ({0}). ' + 
                                           'Die Speicherung und Verarbeitung Ihrer personenbezogenen Daten erfolgt pseudonymisiert<sup>1</sup>, d.h. in namentlich nicht kenntlicher Form, durch {1}. ' + 
                                           'Dies bedeutet, dass die Daten nur mit einem Ihnen zugewiesenen Pseudonym verwendet werden, z.B. VP5 für Versuchsperson Nr. 5. ' + 
                                           'Die Pseudonymisierung der übrigen Daten erfolgt durch die Studienleitenden und deren Stellvertretenden und ist nur diesen bekannt. ' + 
                                           'Weder bei der Erhebung der Daten noch im Rahmen der Auswertung werden Ihr Name oder Ihre Initialen gespeichert. ' + 
                                           '<br /><br /><i><sup>1</sup> Beim Pseudonymisieren erfolgt die Verarbeitung personenbezogener Daten in einer Weise, dass die personenbezogenen Daten ohne Hinzuziehung zusätzlicher Informationen, dem sog. Pseudonymisierungsschlüssel, nicht mehr einer spezifischen betroffenen Person zugeordnet werden können, sofern diese zusätzlichen Informationen gesondert aufbewahrt werden und technischen und organisatorischen Maßnahmen unterliegen, die gewährleisten, dass die personenbezogenen Daten von Unbefugten nicht einer identifizierten oder identifizierbaren natürlichen Person zugewiesen werden können (vgl. DSGVO Artikel 4 Absatz 5)</i>',
            'privacy_datasharing_header': 'Teilen der Daten',
            'privacy_datasharing_declaration': 'Um verbesserte Auswerteprogramme nutzen zu können, werden die Messdaten u.U. in Zusammenarbeit mit Wissenschaftler:innen anderer Arbeitsgruppen ausgewertet, jedoch nur in pseudonymisierter oder anonymisierter<sup>2</sup> Form, sodass andere Wissenschaftler:innen keine Kenntnis davon erhalten, zu welcher Person die analysierten Daten gehören. ' +
                                               '<br /><br /><i><sup>2</sup> Beim Anonymisieren werden die personenbezogenen Daten derart verändert, dass die Einzelangaben über persönliche oder sachliche Verhältnisse nicht mehr oder nur mit unverhältnismäßig großem Aufwand an Zeit, Kosten und Arbeitskraft einer bestimmten oder bestimmbaren natürlichen Person zugeordnet werden können</i>',
            'privacy_datapublication_header': 'Veröffentlichung der Daten',
            'privacy_datapublication_declaration': 'Eine Veröffentlichung von Studienergebnissen erfolgt anonymisiert<sup>2</sup>. Die anonymisierten Daten werden derzeit als „open data“ in einem internetbasierten Forschungsdatenrepositorium namens Open Science Framework ({0}) zugänglich gemacht. Damit folgen wir den Empfehlungen der Deutschen Forschungsgemeinschaft (DFG), die diese Studie mit öffentlichen Mitteln finanziert. Die DFG empfiehlt, dass alle im Rahmen dieser Studie erhobenen Daten der Öffentlichkeit zur Verfügung gestellt werden. Dies ist wichtig um eine Qualitätssicherung in Bezug auf Nachprüfbarkeit und Reproduzierbarkeit<sup>3</sup> wissenschaftlicher Ergebnisse zu gewährleisten und eine optimale Nachnutzung zu erlauben. Dies bezieht sich auch auf die Untersuchung sekundärer Fragestellungen. Die sind zu diesem Zeitpunkt noch nicht bekannt und können daher auch außerhalb der Zweckbestimmung dieser Studie liegen. ' +
                                                   '<br /><br />Gemäß den datenschutzrechtlichen Bestimmungen benötigen wir Ihre Einwilligung zur Speicherung und Verwendung der Daten. ' +
                                                   'Die im Rahmen der Studie erhobenen persönlichen Daten unterliegen der Schweigepflicht und den datenschutzrechtlichen Bestimmungen. ' +
                                                   '<br /><br /><i><sup>3</sup> Reproduzierbarkeit bezieht sich darauf, dass andere Forscher:innen prüfen können ob die Ergebnisse korrekt sind und ob sie zu identischen Ergebnissen kommen würden</i>',
            'privacy_voluntary_header': '2. Freiwilligkeit/Studienabbruch',
            'privacy_voluntary_declaration': 'Die Teilnahme an dieser Studie ist absolut freiwillig. Durch Ihre Bereitschaft, an dieser Studie teilzunehmen, unterstützen Sie maßgeblich die Erforschung von Wahrnehmungsprozessen.',
            'privacy_additionalrights_header': 'Außerdem haben Sie jeweils das Recht',
            'privacy_additionalrights_declaration': ulist(
                                                        '&nbsp;auf Auskunft über alle zu Ihrer Person verarbeiteten und gespeicherten Daten sowie der Empfänger, an die Daten weitergegeben werden oder wurden, Art. 15 DS-GVO;',
                                                        '&nbsp;auf Berichtigung unrichtiger personenbezogener Daten, Art. 16 DS-GVO;', 
                                                        '&nbsp;der Weiterverarbeitung Ihrer personenbezogenen Daten zu widersprechen, die ohne Ihre Einwilligung aufgrund eines öffentlichen Interesses oder zur Wahrung berechtigter Interessen des/der Verantwortlichen erfolgt ist. Der Widerspruch einer Weiterverarbeitung ist zu begründen, sodass deutlich wird, dass besondere in Ihrer Person begründete Umstände das vorgenannte Interesse an einer Weiterverarbeitung überwiegen, Art. 21 DS-GVO;', 
                                                        '&nbsp;auf Löschung unter der Voraussetzung, dass bestimmte Gründe vorliegen. Dies ist insbesondere der Fall bei unrechtmäßiger Verarbeitung oder wenn die Daten zu dem Zweck, zu dem sie erhoben oder verarbeitet wurden, nicht mehr notwendig sind, Sie die Einwilligung widerrufen und eine anderweitige Rechtsgrundlage für die Datenverarbeitung nicht gegeben ist oder anstelle des vorbenannten Widerspruchs nach Art. 21 DS-GVO unter den dort genannten Voraussetzungen. Sofern die Löschung die Ziele eines im wissenschaftlichen Interesse durchgeführten Forschungsprojektes zunichtemachen oder wesentlich erschweren würde, besteht kein Recht auf Löschen, Art. 17 Absatz 3 DS-GVO. Nach Ablauf einer Aufbewahrungszeit von 10 Jahren werden Ihre personenbezogenen Daten gelöscht.',
                                                        '&nbsp;auf Einschränkung der Verarbeitung Ihrer personenbezogenen Daten, insbesondere wenn die Verarbeitung unrechtmäßig ist und Sie die Einschränkung anstelle des Löschens verlangen (siehe dort) oder solange streitig ist, ob die Verarbeitung personenbezogener Daten rechtmäßig erfolgt, Art. 18 DS-GVO.'
                                                    ),
            'privacy_compensation_header': '3. Aufwandsentschädigung',
            'privacy_compensation_declaration': 'Das Experiment wird insgesamt etwa {0} Minuten dauern. Es wird eine Aufwandsentschädigung von {1} pro Stunde nach Abschluss der Untersuchung bezahlt.',
            'privacy_contact_header': '4. Kontaktdaten Datenschutzbeauftragte:r und Datenschutzaufsichtsbehörde',
            'privacy_contact_declaration': 'Verantwortlich für den Datenschutz ist {0} ({1}), {2}, Tel. {3}, E-Mail: {4}. ' + 
                                           'Es wird auf ein Beschwerderecht bei der lokalen Datenschutzaufsichtsbehörde Hamburg, Kurt-Schumacher-Allee 4, 20097 Hamburg, (040) 42854–4040, mailbox@datenschutz.hamburg.de hingewiesen. Weitere Informationen sind im Internet verfügbar unter {5}. Sollten Sie Fragen zur Datenverarbeitung haben, können Sie sich für weitere Auskünfte an den Datenschutzbeauftragten des UKE wenden: Matthias Jaster, Martinistraße 52, 20246 Hamburg, Tel. (040) 7410-56890, E-Mail: dsb@uke.de.',
            'privacy_ethics_header': 'Allgemeine Hinweise',
            'privacy_ethics_declaration': 'Diese Studie ist von der unabhängigen Ethikkommission der Ärztekammer Hamburg hinsichtlich ihrer medizinischen, rechtlichen und ethischen Vertretbarkeit beraten worden. Die Verantwortung für die Durchführung verbleibt jedoch bei den Studienleitenden.',
            
            // text for consent form
            'consent_header': 'Einwilligung in die Verarbeitung personenbezogenen Daten',
            'consent_declaration': 'Mir ist bekannt, dass bei dieser Studie personenbezogene Daten verarbeitet werden sollen. Die Verarbeitung der Daten setzt gemäß der Datenschutz-Grundverordnung (DS-GVO) die Abgabe folgender Einwilligungserklärung voraus:<br /><br />' + 
                                   'Ich wurde ausführlich und verständlich darüber aufgeklärt, dass meine in der Studie erhobenen Daten in pseudonymisierter Form gespeichert und ausgewertet werden. Mir ist bekannt, dass eine Rückverfolgung der Datenverarbeitung ausgeschlossen ist, sodass ich meine Rechte auf Auskunft, Berichtigung oder Löschung nicht mehr durchsetzen kann.<br /><br />' + 
                                   'Außerdem kann ich Beschwerde bei einer Datenschutzbehörde einlegen.<br /><br />' + 
                                   'Der Anonymisierung meiner personenbezogenen Daten zum Zwecke der Veröffentlichung oder Weitergabe an Kooperationspartner:innen stimme ich zu.<br /><br />' + 
                                   '<div id="consent-tooltip"><i>Eine der Antwortmöglichkeiten muss ausgewählt werden.</i></div>',
            'consent_choice_yes': 'Ja, ich möchte an der Befragung teilnehmen, bin über 18 Jahre alt und bin damit einverstanden, dass meine Daten in anonymisierter Form für wissenschaftliche Zwecke verwendet werden.',
            'consent_choice_no': 'Nein, ich möchte an der Studie nicht teilnehmen.',
            'consent_submit': 'Bestätigen',
            'consent_rejected': 'Sie haben der Einwilligungserklärung nicht zugestimmt, daher ist das Experiment an dieser Stelle beendet. Bei Fragen richten Sie sich ggf. bitte über Prolific an die Experimentleitung.                                                            '
        },
        'en': {
            // navigation text
            'navigation_previous': 'Previous',
            'navigation_continue': 'Continue',

            // text for general briefing
            'general_header': 'Information, briefing, consent',
            'general_greeting': 'Dear participant,',
            'general_introduction': 'As part of an ongoing research project conducted by {0}, we are investigating {1}.',
            'general_procedure': 'Procedure',
            'general_inclusion': 'Please participate only if you meet all of the following inclusion criteria:',
            'general_exclusion': 'Please do <b>not</b> participate if you meet any of the following exclusion criteria:',
            'general_signoff': 'We would be very grateful for your participation. Thank you!',
            'general_and': 'and',

            // text for privacy briefing
            'privacy_header': 'Information, privacy, consent',
            'privacy_storage_header': '1. Information on the processing of anonymised data',
            'privacy_storage_declaration': 'This online experiment is conducted via Pavlovia. Pavlovia is compliant with the EU General Data Protection Regulation ({0}). ' +
                                        'The storage and processing of your personal data are carried out in pseudonymised<sup>1</sup> form, that is, in a form in which your name cannot be identified, by {1}. ' +
                                        'This means that the data are only used with a pseudonym assigned to you, for example VP5 for participant no. 5. ' +
                                        'The pseudonymisation of the remaining data is carried out by the principal investigators and their deputies and is known only to them. ' +
                                        'Neither during data collection nor during data analysis will your name or initials be stored. ' +
                                        '<br /><br /><i><sup>1</sup> In pseudonymisation, personal data are processed in such a way that the personal data can no longer be attributed to a specific data subject without the use of additional information, the so called pseudonymisation key, provided that such additional information is kept separately and is subject to technical and organisational measures ensuring that the personal data cannot be assigned to an identified or identifiable natural person by unauthorised persons (cf. GDPR Article 4(5)).</i>',
            'privacy_datasharing_header': 'Data sharing',
            'privacy_datasharing_declaration': 'In order to be able to use advanced analysis software, the measurement data may, where applicable, be analysed in collaboration with researchers from other working groups, but only in pseudonymised or anonymised<sup>2</sup> form, so that other researchers do not obtain any information about which person the analysed data belong to. ' +
                                            '<br /><br /><i><sup>2</sup> In anonymisation, personal data are altered in such a way that individual details about personal or factual circumstances can no longer, or only with a disproportionate amount of time, cost and effort, be attributed to a specific or identifiable natural person.</i>',
            'privacy_datapublication_header': 'Publication of data',
            'privacy_datapublication_declaration': 'The publication of study results will be in anonymised<sup>2</sup> form. The anonymised data are currently made available as open data in an internet based research data repository called Open Science Framework ({0}). In doing so we follow the recommendations of the German Research Foundation (DFG), which funds this study with public money. The DFG recommends that all data collected as part of this study be made available to the public. This is important in order to ensure quality control with regard to the verifiability and reproducibility<sup>3</sup> of scientific findings and to allow optimal reuse. This also applies to the investigation of secondary research questions. These are not yet known at this time and may therefore also go beyond the original purpose of this study. ' +
                                            '<br /><br />In accordance with data protection regulations, we require your consent to store and use the data. ' +
                                            'The personal data collected as part of the study are subject to confidentiality and data protection regulations. ' +
                                            '<br /><br /><i><sup>3</sup> Reproducibility refers to the ability of other researchers to verify whether the results are correct and whether they would arrive at identical results.</i>',
            'privacy_voluntary_header': '2. Voluntary participation / withdrawal from the study',
            'privacy_voluntary_declaration': 'Participation in this study is entirely voluntary. By agreeing to take part in this study, you make an important contribution to research on perceptual processes.',
            'privacy_additionalrights_header': 'In addition, you have the right',
            'privacy_additionalrights_declaration': ulist(
                '&nbsp;to obtain information about all data relating to you that are processed and stored, as well as about the recipients to whom the data are or have been disclosed, Art. 15 GDPR;',
                '&nbsp;to rectification of inaccurate personal data, Art. 16 GDPR;',
                '&nbsp;to object to the further processing of your personal data that has taken place without your consent on the basis of a public interest or for the purposes of legitimate interests pursued by the controller. The objection to further processing must be justified so that it becomes clear that special circumstances relating to your person outweigh the above mentioned interest in further processing, Art. 21 GDPR;',
                '&nbsp;to erasure, provided that certain grounds apply. This is in particular the case where processing is unlawful or where the data are no longer necessary for the purpose for which they were collected or processed, where you withdraw your consent and there is no other legal basis for the data processing, or instead of the aforementioned objection under Art. 21 GDPR under the conditions set out there. If erasure would render impossible or seriously impair the achievement of the objectives of a research project carried out in the public interest, there is no right to erasure, Art. 17(3) GDPR. After a retention period of 10 years your personal data will be erased.',
                '&nbsp;to restriction of the processing of your personal data, in particular where the processing is unlawful and you request restriction instead of erasure (see above), or for as long as it is disputed whether the processing of personal data is lawful, Art. 18 GDPR.'
            ),
            'privacy_compensation_header': '3. Compensation',
            'privacy_compensation_declaration': 'The experiment will take approximately {0} minutes in total. You will receive compensation of {1} per hour after completion of the study.',
            'privacy_contact_header': '4. Contact details of the data protection officer and the data protection supervisory authority',
            'privacy_contact_declaration': 'The person responsible for data protection is {0} ({1}), {2}, tel. {3}, e mail: {4}. ' +
                                        'Reference is made to the right to lodge a complaint with the local data protection supervisory authority in Hamburg, Kurt Schumacher Allee 4, 20097 Hamburg, (040) 42854–4040, mailbox@datenschutz.hamburg.de. Further information is available on the internet at {5}. If you have any questions regarding data processing, you may contact the data protection officer of the UKE for further information: Matthias Jaster, Martinistraße 52, 20246 Hamburg, tel. (040) 7410-56890, e mail: dsb@uke.de.',
            'privacy_ethics_header': 'General information',
            'privacy_ethics_declaration': 'This study has been reviewed by the independent ethics committee of the Hamburg Chamber of Physicians with regard to its medical, legal and ethical acceptability. Responsibility for conducting the study nevertheless remains with the principal investigators.',

            // text for consent form
            'consent_header': 'Consent to the processing of personal data',
            'consent_declaration': 'I am aware that personal data will be processed in this study. According to the General Data Protection Regulation (GDPR), the processing of the data requires the following declaration of consent:<br /><br />' +
                                'I have been informed comprehensively and in an understandable manner that the data collected about me in the study will be stored and analysed in pseudonymised form. I am aware that it will no longer be possible to trace the data processing back to me, so that I can no longer exercise my rights of access, rectification or erasure.<br /><br />' +
                                'I can also lodge a complaint with a data protection authority.<br /><br />' +
                                'I consent to the anonymisation of my personal data for the purpose of publication or transfer to cooperation partners.<br /><br />' +
                                '<div id="consent-tooltip"><i>One of the response options must be selected.</i></div>',
            'consent_choice_yes': 'Yes, I would like to take part in the study, I am over 18 years of age and I agree that my data may be used in anonymised form for scientific purposes.',
            'consent_choice_no': 'No, I do not wish to take part in the study.',
            'consent_submit': 'Confirm',
            'consent_rejected': 'You have not agreed to the declaration of consent, so the experiment ends at this point. If you have any questions, please contact the experimenter via Prolific.                                                            '
        }
    };

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: 'informed_consent_uke',
        parameters: {
            study_language: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'de',
                pretty_name: "Study language",
                description: "Language to be used during the study. Available options are: 'de'."
            },
            study_title: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Study title",
                description: "Title of the study, to be displayed in the brief."
            },
            study_subtitle: {
                type: jsPsych.plugins.parameterType.STRING,
                default: "",
                pretty_name: "Study subtitle",
                description: "Subtitle of the study, to be displayed in the brief."
            },
            study_centre: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Study centre",
                description: "Name of the centre hosting the study, to be displayed in the brief."
            },
            study_purpose: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Study purpose",
                description: "The purpose of the study, to be displayed in the brief."
            },
            study_description: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Study description",
                description: "Short description of the study, to be displayed in the brief. This should include, at a minimum, the duration of the experiment and a one-line explanation of the task."
            },
            study_inclusion_criteria: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Inclusion criteria",
                description: "A list of inclusion criteria. Note that this should be supplied as html. Helper functions are provided (see `ulist`)."
            },
            study_exclusion_criteria: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Exclusion criteria",
                description: "A list of exclusion criteria. Note that this should be supplied as html. Helper functions are provided (see `ulist`)."
            },
            study_contact_person: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Contact person",
                description: "The person responsible for the study."
            },
            study_duration_in_minutes: {
                type: jsPsych.plugins.parameterType.INT,
                default: undefined,
                pretty_name: "Study duration",
                description: "The duration of the study, in minutes."
            },
            study_hourly_rate: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Hourly pay",
                description: "The hourly rate that is paid to participants. Note that this should be supplied as a string that includes the currency."
            },
            study_privacy_contact_person: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Privacy contact person",
                description: "The person resposible for privacy concerns for the study."
            },
            study_privacy_contact_centre: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Privacy contact person's centre",
                description: "Affiliation of the privacy contact person."
            },
            study_privacy_contact_address: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Privacy contact address",
                description: "Address of the privacy contact person."
            },
            study_privacy_contact_phone: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Privacy contact phone",
                description: "Phone number of the privacy contact person."
            },
            study_privacy_contact_email: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Privacy contact email",
                description: "eMail address of the privacy contact person."
            },
            f_callback: {
                type: jsPsych.plugins.parameterType.FUNCTION,
                default: function(e){},
                pretty_name: "Callback function",
                description: "Callback for when participants fail to comply."
            }
        }
    }

    // trial logic
    plugin.trial = function(e, t) {
        // log start of trial
        console.log("Trial", t, ": Starting...");

        // setup data structure
        let data = { 
            consent_given: false,
            consent_timestamp: null
        };

        // setup brief
        let pBriefing = function(){
            // create the main container
            let trialContainer = $(
                '<div id="trial-header" style="width: 100%; font-size: small;" align="center">' + 
                    '<div id="trial-header-line1" style="width: 75%; display: flex; justify-content: space-between;" align="justify"></div>' + 
                    '<div style="width: 75%;" align="justify">' + t.study_subtitle + '</div>' + 
                '</div>' + 
                '<div id="trial-container" style="width: 100%; max-height:70vh; overflow-y:scroll; font-size: small; background: #eee;" align="center">' + 
                    '<div id="brief-container" style="width: 75%;" align="justify"></div>' + 
                '</div>' + 
                '<formset>' + 
                    '<button id="trial-previous" type="button">' + lang[t.study_language].navigation_previous + '</button>' + 
                    '&nbsp;&nbsp;&nbsp;' + 
                    '<button id="trial-continue" type="button">' + lang[t.study_language].navigation_continue + '</button>' + 
                '</formset>'
            );

            // get handle of header and brief containers
            let headerContainer = trialContainer.find('#trial-header-line1');
            let briefContainer = trialContainer.find('#brief-container');

            // get handles of buttons
            let buttonPrevious = trialContainer.find('#trial-previous');
            let buttonContinue = trialContainer.find('#trial-continue');

            // create general header
            let general_header = $(
                paragraphs(
                    bold(lang[t.study_language].general_header),
                    italic(t.study_title)
                )
            );
            
            // create general brief
            let general = $(
                paragraphs(
                    lang[t.study_language].general_greeting,
                    lang[t.study_language].general_introduction.format(t.study_centre, t.study_purpose),
                    bold(lang[t.study_language].general_procedure) + '<br />' + t.study_description,
                    bold(lang[t.study_language].general_inclusion) + t.study_inclusion_criteria,
                    bold(lang[t.study_language].general_exclusion) + t.study_exclusion_criteria,
                    lang[t.study_language].general_signoff,
                    t.study_contact_person + '<br />' + lang[t.study_language].general_and + ' ' + t.study_centre
                )
            );

            // create privacy header
            let privacy_header = $(
                paragraphs(
                    bold(lang[t.study_language].privacy_header),
                    italic(t.study_title)
                )
            );
            
            // create privacy brief
            let privacy = $(
                paragraphs(
                    bold(lang[t.study_language].privacy_storage_header),
                    lang[t.study_language].privacy_storage_declaration.format('<a href="https://pavlovia.org/docs/home/ethics" target="_blank">https://pavlovia.org/docs/home/ethics</a>', t.study_centre),
                    bold(lang[t.study_language].privacy_datasharing_header),
                    lang[t.study_language].privacy_datasharing_declaration,
                    bold(lang[t.study_language].privacy_datapublication_header),
                    lang[t.study_language].privacy_datapublication_declaration.format('<a href="https://osf.io" target="_blank">https://osf.io</a>'),
                    bold(lang[t.study_language].privacy_voluntary_header),
                    lang[t.study_language].privacy_voluntary_declaration,
                    bold(lang[t.study_language].privacy_additionalrights_header),
                    lang[t.study_language].privacy_additionalrights_declaration,
                    bold(lang[t.study_language].privacy_compensation_header),
                    lang[t.study_language].privacy_compensation_declaration.format(t.study_duration_in_minutes.toString(), t.study_hourly_rate),
                    bold(lang[t.study_language].privacy_contact_header),
                    lang[t.study_language].privacy_contact_declaration.format(t.study_privacy_contact_person, t.study_privacy_contact_centre, t.study_privacy_contact_address, t.study_privacy_contact_phone, t.study_privacy_contact_email, '<a href="https://datenschutz-hamburg.de" target="_blank">https://datenschutz-hamburg.de</a>'),
                    bold(lang[t.study_language].privacy_ethics_header), 
                    lang[t.study_language].privacy_ethics_declaration
                )
            );

            // setup a clear function
            var clearEverything = function(){
                e.innerHTML = '';
                general_header.hide();
                general.hide();
                privacy_header.hide();
                privacy.hide();
                buttonPrevious.hide();
                buttonContinue.hide();
                buttonPrevious.off('click');
                buttonContinue.off('click');
            }

            // setup handling of general call
            var handleGeneral = function(){
                // clean up
                clearEverything();

                // show general brief
                trialContainer.appendTo(e);
                general_header.show();
                general.show();

                // enable continue button
                buttonContinue.show();
                buttonContinue.on('click', handlePrivacy);
            }

            // setup handling of privacy call
            var handlePrivacy = function(){
                // clean up
                clearEverything();

                // show privacy brief
                trialContainer.appendTo(e);
                privacy_header.show();
                privacy.show();

                // enable continue button
                buttonContinue.show();
                buttonContinue.on('click', function(){
                    // clean up
                    clearEverything();

                    // move on
                    pConsentForm();
                });

                // enable previous button
                buttonPrevious.show();
                buttonPrevious.on('click', handleGeneral);
            }

            // append fields once
            general_header.appendTo(headerContainer);
            general.appendTo(briefContainer);
            privacy_header.appendTo(headerContainer);
            privacy.appendTo(briefContainer);

            // start process
            handleGeneral();
        }

        // setup consent form
        let pConsentForm = function(){
            // create the main form
            let form = $(
                '<div id="trial-header" style="width: 100%; font-size: small;" align="center">' + 
                    '<div id="trial-header-line1" style="width: 75%; display: flex; justify-content: space-between;" align="justify">' + 
                        paragraphs(
                            bold(lang[t.study_language].consent_header),
                            italic(t.study_title)
                        ) + 
                    '</div>' + 
                    '<div style="width: 75%;" align="justify">' + t.study_subtitle + '</div>' + 
                '</div>' + 
                '<div id="trial-container" style="width: 100%; max-height:70vh; overflow-y:scroll; font-size: small; background: #eee;" align="center">' + 
                    '<div style="width: 75%;" align="justify">' + 
                        paragraphs(
                            lang[t.study_language].consent_declaration,
                            '<fieldset id="consent-form">' + 
                                '<input type="radio" id="yes" name="consent" value="yes"><label for="yes"> ' + lang[t.study_language].consent_choice_yes + '</label><br />' +
                                '<input type="radio" id="no" name="consent" value="no"><label for="no"> ' + lang[t.study_language].consent_choice_no + '</label><br />' +
                                '<button type="button" id="consent-submit">' + lang[t.study_language].consent_submit + '</button>' + 
                            '</fieldset>'
                        ) +
                    '</div>' + 
                '</div>'
            );

            // create the rejection screen
            let rejection = $(
                '<div id="trial-header" style="width: 100%; font-size: small;" align="center">' + 
                    '<div id="trial-header-line1" style="width: 75%; display: flex; justify-content: space-between;" align="justify">' + 
                        paragraphs(
                            bold(lang[t.study_language].consent_header),
                            italic(t.study_title)
                        ) + 
                    '</div>' + 
                    '<div style="width: 75%;" align="justify">' + t.study_subtitle + '</div>' + 
                '</div>' + 
                '<div id="trial-container" style="width: 100%; font-size: small; background: #eee;" align="center">' + 
                    '<div style="width: 75%;" align="justify">' + 
                        paragraphs(
                            lang[t.study_language].consent_rejected + 
                            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 
                            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 
                            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 
                            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 
                            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 
                            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                        ) +
                    '</div>' + 
                '</div>'
            );
            
            // grab relevant handles
            let fieldset = form.find('#consent-form');
            let tooltip = form.find('#consent-tooltip');
            let submit = form.find('#consent-submit');

            // add event listener
            submit.on('click', function(){
                // read consent choice
                let choice = $("input[name=consent]:checked", '#consent-form').val();

                // tooltip if unchecked
                if (typeof(choice) === 'undefined') {
                    fieldset.css({'border': '1px solid red'});
                    tooltip.show();
                    return;
                }

                // remove listener
                $(this).off('click');

                // abort experiment if consent is not given
                if (choice == 'no') {
                    console.log("Experiment aborted due to lack of consent.");

                    jsPsych.pluginAPI.clearAllTimeouts();
                    jsPsych.pluginAPI.cancelAllKeyboardResponses();

                    e.innerHTML = '';
                    rejection.appendTo(e);

                    if (typeof(t.f_callback) !== 'undefined') {
                        t.f_callback(e);
                    }

                    return;
                }

                // set consent
                data.consent_given = true;
                data.consent_timestamp = + new Date() ? Date.now() : new Date().getTime();

                // finish trial
                jsPsych.pluginAPI.clearAllTimeouts();
                jsPsych.finishTrial(data);
            });

            // cosmetics
            fieldset.css({'border': '1px solid gray'})
            tooltip.hide();

            e.innerHTML = '';
            form.appendTo(e);
        }

        // start process
        pBriefing();
    }

    return plugin;
})();