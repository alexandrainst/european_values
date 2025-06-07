"""Constants used in the project."""

COUNTRY_GROUPS = {
    "Caribbean": [
        "AG",  # Antigua and Barbuda
        "BB",  # Barbados
        "BS",  # Bahamas
        "CU",  # Cuba
        "DM",  # Dominica
        "DO",  # Dominican Republic
        "GD",  # Grenada
        "GF",  # French Guiana
        "GY",  # Guyana
        "HT",  # Haiti
        "JM",  # Jamaica
        "KN",  # Saint Kitts and Nevis
        "LC",  # Saint Lucia
        "PR",  # Puerto Rico
        "SR",  # Suriname
        "TT",  # Trinidad and Tobago
        "VC",  # Saint Vincent and the Grenadines
    ],
    "Central America": [
        "BZ",  # Belize
        "CR",  # Costa Rica
        "GT",  # Guatemala
        "HN",  # Honduras
        "MX",  # Mexico
        "NI",  # Nicaragua
        "PA",  # Panama
        "SV",  # El Salvador
    ],
    "Central Asia": [
        "KG",  # Kyrgyzstan
        "KZ",  # Kazakhstan
        "TJ",  # Tajikistan
        "TM",  # Turkmenistan
        "UZ",  # Uzbekistan
    ],
    "Central Europe": [
        "CZ",  # Czech Republic
        "HU",  # Hungary
        "PL",  # Poland
        "SI",  # Slovenia
        "SK",  # Slovakia
    ],
    "China": [
        "CN"  # China
    ],
    "East Asia Without China": [
        "HK",  # Hong Kong
        "JP",  # Japan
        "KP",  # North Korea
        "KR",  # South Korea
        "MN",  # Mongolia
        "MO",  # Macao
        "TW",  # Taiwan
    ],
    "Eastern Europe": [
        "AL",  # Albania
        "AM",  # Armenia
        "AZ",  # Azerbaijan
        "BA",  # Bosnia and Herzegovina
        "BG",  # Bulgaria
        "BY",  # Belarus
        "GE",  # Georgia
        "HR",  # Croatia
        "MD",  # Moldova
        "ME",  # Montenegro
        "MK",  # North Macedonia
        "RO",  # Romania
        "RS",  # Serbia
        "RU",  # Russia
        "TR",  # Turkey
        "UA",  # Ukraine
        "XK",  # Kosovo
    ],
    "Middle East": [
        "AE",  # United Arab Emirates
        "BH",  # Bahrain
        "CY",  # Cyprus
        "EG",  # Egypt
        "IL",  # Israel
        "IQ",  # Iraq
        "IR",  # Iran
        "JO",  # Jordan
        "KW",  # Kuwait
        "LB",  # Lebanon
        "OM",  # Oman
        "PS",  # Palestine, State of (or Palestinian Territories)
        "QA",  # Qatar
        "SA",  # Saudi Arabia
        "SY",  # Syria
        "TR",  # Turkey
        "YE",  # Yemen
    ],
    "North Africa": [
        "DZ",  # Algeria
        "EG",  # Egypt
        "LY",  # Libya
        "MA",  # Morocco
        "MR",  # Mauritania
        "SD",  # Sudan
        "TN",  # Tunisia
    ],
    "North America": [
        "CA",  # Canada
        "US",  # United States
    ],
    "Northern Europe": [
        "DK",  # Denmark
        "EE",  # Estonia
        "FI",  # Finland
        "FO",  # Faroe Islands
        "GL",  # Greenland
        "IS",  # Iceland
        "LT",  # Lithuania
        "LV",  # Latvia
        "NO",  # Norway
        "SE",  # Sweden
    ],
    "Oceania": [
        "AS",  # American Samoa
        "AU",  # Australia
        "CK",  # Cook Islands
        "FJ",  # Fiji
        "FM",  # Micronesia
        "GU",  # Guam
        "KI",  # Kiribati
        "MH",  # Marshall Islands
        "MP",  # Northern Mariana Islands
        "NC",  # New Caledonia
        "NR",  # Nauru
        "NU",  # Niue
        "NZ",  # New Zealand
        "PF",  # French Polynesia
        "PG",  # Papua New Guinea
        "PN",  # Pitcairn Islands
        "PW",  # Palau
        "SB",  # Solomon Islands
        "TK",  # Tokelau
        "TO",  # Tonga
        "TV",  # Tuvalu
        "VU",  # Vanuatu
        "WF",  # Wallis and Futuna
        "WS",  # Samoa
    ],
    "South America": [
        "AR",  # Argentina
        "BO",  # Bolivia
        "BR",  # Brazil
        "CL",  # Chile
        "CO",  # Colombia
        "EC",  # Ecuador
        "PE",  # Peru
        "PY",  # Paraguay
        "UY",  # Uruguay
        "VE",  # Venezuela
    ],
    "South Asia": [
        "AF",  # Afghanistan
        "BD",  # Bangladesh
        "BT",  # Bhutan
        "IN",  # India
        "LK",  # Sri Lanka
        "MV",  # Maldives
        "NP",  # Nepal
        "PK",  # Pakistan
    ],
    "Southeast Asia": [
        "BN",  # Brunei Darussalam
        "ID",  # Indonesia
        "KH",  # Cambodia
        "LA",  # Laos
        "MM",  # Myanmar
        "MY",  # Malaysia
        "PH",  # Philippines
        "SG",  # Singapore
        "TH",  # Thailand
        "TL",  # Timor-Leste
        "VN",  # Vietnam
    ],
    "Southern Europe": [
        "AD",  # Andorra
        "CY",  # Cyprus
        "ES",  # Spain
        "GR",  # Greece
        "IT",  # Italy
        "MT",  # Malta
        "PT",  # Portugal
        "SM",  # San Marino
        "VA",  # Vatican City
    ],
    "Sub Saharan Africa": [
        "AO",  # Angola
        "BF",  # Burkina Faso
        "BI",  # Burundi
        "BJ",  # Benin
        "BW",  # Botswana
        "CD",  # Congo (Democratic Republic of the)
        "CF",  # Central African Republic
        "CG",  # Congo (Republic of the)
        "CI",  # Côte d'Ivoire
        "CM",  # Cameroon
        "CV",  # Cape Verde
        "DJ",  # Djibouti
        "ER",  # Eritrea
        "ET",  # Ethiopia
        "GA",  # Gabon
        "GH",  # Ghana
        "GM",  # Gambia
        "GN",  # Guinea
        "GQ",  # Equatorial Guinea
        "GW",  # Guinea-Bissau
        "KE",  # Kenya
        "LS",  # Lesotho
        "LR",  # Liberia
        "MG",  # Madagascar
        "ML",  # Mali
        "MW",  # Malawi
        "MZ",  # Mozambique
        "NA",  # Namibia
        "NE",  # Niger
        "NG",  # Nigeria
        "RW",  # Rwanda
        "SC",  # Seychelles
        "SL",  # Sierra Leone
        "SO",  # Somalia
        "SS",  # South Sudan
        "ST",  # Sao Tome and Principe
        "SZ",  # Eswatini
        "TD",  # Chad
        "TG",  # Togo
        "TZ",  # Tanzania
        "UG",  # Uganda
        "ZA",  # South Africa
        "ZM",  # Zambia
        "ZW",  # Zimbabwe
    ],
    "Western Europe": [
        "AT",  # Austria
        "BE",  # Belgium
        "CH",  # Switzerland
        "DE",  # Germany
        "FR",  # France
        "GB",  # United Kingdom
        "IE",  # Ireland
        "LI",  # Liechtenstein
        "LU",  # Luxembourg
        "MC",  # Monaco
        "NL",  # Netherlands
        "NIR",  # Northern Ireland
    ],
}


EVS_TREND_ANSWER_COLUMNS = dict(
    A001="question_a001_important_in_life__family",
    A002="question_a002_important_in_life__friends",
    A003="question_a003_important_in_life__leisure_time",
    A004="question_a004_important_in_life__politics",
    A005="question_a005_important_in_life__work",
    A006="question_a006_important_in_life__religion",
    A007="question_a007_important_in_life__service_to_others",
    A008="question_a008_feeling_of_happiness",
    A009="question_a009_state_of_health__subjective",
    A010="question_a010_ever_felt_very_excited_or_interested",
    A011="question_a011_ever_felt_restless",
    A012="question_a012_ever_felt_proud_because_someone_complimented_you",
    A013="question_a013_ever_felt_very_lonely_or_remote_from_other_people",
    A014="question_a014_ever_felt_pleased_about_having_accomplished_something",
    A015="question_a015_ever_felt_bored",
    A016="question_a016_ever_felt_on_top_of_the_world",
    A017="question_a017_ever_felt_depressed_or_very_unhappy",
    A018="question_a018_ever_felt_that_things_were_going_your_way",
    A019="question_a019_ever_felt_upset_because_somebody_criticized_you",
    A025="question_a025_respect_and_love_for_parents",
    A026="question_a026_parents_responsibilities_to_their_children",
    A027="question_a027_important_child_qualities__good_manners",
    A029="question_a029_important_child_qualities__independence",
    A030="question_a030_important_child_qualities__hard_work",
    A032="question_a032_important_child_qualities__feeling_of_responsibility",
    A034="question_a034_important_child_qualities__imagination",
    A035="question_a035_important_child_qualities__tolerance_and_respect_for_other_people",
    A038="question_a038_important_child_qualities__thrift_saving_money_and_things",
    A039="question_a039_important_child_qualities__determination_perseverance",
    A040="question_a040_important_child_qualities__religious_faith",
    A041="question_a041_important_child_qualities__unselfishness",
    A042="question_a042_important_child_qualities__obedience",
    A043_01="question_a043_01_important_child_qualities__none",
    A046="question_a046_abortion_when_the_mothers_health_is_at_risk",
    A047="question_a047_abortion_when_child_physically_handicapped",
    A048="question_a048_abortion_when_woman_not_married",
    A049="question_a049_abortion_if_not_wanting_more_children",
    A058="question_a058_spend_time_with_friends",
    A059="question_a059_spend_time_with_colleagues_from_work",
    A060="question_a060_spend_time_with_people_at_your_church_mosque_or_synagogue",
    A061="question_a061_spend_time_with_people_at_sport_culture_communal_organization",
    A062="question_a062_how_often_discusses_political_matters_with_friends",
    A063="question_a063_persuading_friends_relatives_or_fellow_workers",
    A064="question_a064_member__belong_to_social_welfare_service_for_elderly_handicapped_or_deprived_people",
    A065="question_a065_member__belong_to_religious_organization",
    A066="question_a066_member__belong_to_education_arts_music_or_cultural_activities",
    A067="question_a067_member__belong_to_labour_unions",
    A068="question_a068_member__belong_to_political_parties",
    A069="question_a069_member__belong_to_local_political_actions",
    A070="question_a070_member__belong_to_human_rights",
    A071="question_a071_member__belong_to_conservation_the_environment_ecology_animal_rights",
    A071B="question_a071b_member__belong_to_conservation_the_environment_ecology",
    A071C="question_a071c_member__belong_to_animal_rights",
    A072="question_a072_member__belong_to_professional_associations",
    A073="question_a073_member__belong_to_youth_work",
    A074="question_a074_member__belong_to_sports_or_recreation",
    A075="question_a075_member__belong_to_women’s_group",
    A076="question_a076_member__belong_to_peace_movement",
    A077="question_a077_member__belong_to_organization_concerned_with_health",
    A078="question_a078_member__belong_to_consumer_groups",
    A079="question_a079_member__belong_to_other_groups",
    A080="question_a080_member__belong_to_none",
    A080_01="question_a080_01_member__belong_to_humanitarian_or_charitable_organization",
    A080_02="question_a080_02_member__belong_to_self_help_group_mutual_aid_group",
    A081="question_a081_voluntary_work__unpaid_work_social_welfare_service_for_elderly_handicapped_or_deprived_people",
    A082="question_a082_voluntary_work__unpaid_work_religious_or_church_organization",
    A083="question_a083_voluntary_work__unpaid_work_education_arts_music_or_cultural_activities",
    A084="question_a084_voluntary_work__unpaid_work_labour_unions",
    A085="question_a085_voluntary_work__unpaid_work_political_parties_or_groups",
    A086="question_a086_voluntary_work__unpaid_work_local_political_action_groups",
    A087="question_a087_voluntary_work__unpaid_work_human_rights",
    A088="question_a088_voluntary_work__unpaid_work_environment_conservation_animal_rights",
    A088B="question_a088b_voluntary_work__unpaid_work_environment_conservation_ecology",
    A088C="question_a088c_voluntary_work__unpaid_work_animal_rights",
    A089="question_a089_voluntary_work__unpaid_work_professional_associations",
    A090="question_a090_voluntary_work__unpaid_work_youth_work",
    A091="question_a091_voluntary_work__unpaid_work_sports_or_recreation",
    A092="question_a092_voluntary_work__unpaid_work_women’s_group",
    A093="question_a093_voluntary_work__unpaid_work_peace_movement",
    A094="question_a094_voluntary_work__unpaid_work_organization_concerned_with_health",
    A096="question_a096_voluntary_work__unpaid_work_other_groups",
    A097="question_a097_voluntary_work__unpaid_work_none",
    A107="question_a107_reasons_voluntary_work__solidarity_with_the_poor_and_disadvantaged",
    A108="question_a108_reasons_voluntary_work__compassion_for_those_in_need",
    A109="question_a109_reasons_voluntary_work__opportunity_to_repay_something",
    A110="question_a110_reasons_voluntary_work__sense_of_duty_moral_obligation",
    A111="question_a111_reasons_voluntary_work__identifying_with_people_who_suffer",
    A112="question_a112_reasons_voluntary_work__time_on_my_hands",
    A113="question_a113_reasons_voluntary_work__personal_satisfaction",
    A114="question_a114_reasons_voluntary_work__religious_belief",
    A115="question_a115_reasons_voluntary_work__help_disadvantaged_people",
    A116="question_a116_reasons_voluntary_work__make_a_contribution_to_my_local_community",
    A117="question_a117_reasons_voluntary_work__bring_about_social_or_political_change",
    A118="question_a118_reasons_voluntary_work__for_social_reasons",
    A119="question_a119_reasons_voluntary_work__gain_new_skills_and_useful_experience",
    A120="question_a120_reasons_voluntary_work__did_not_want_to_but_could_not_refuse",
    A124_01="question_a124_01_neighbours__people_with_a_criminal_record",
    A124_02="question_a124_02_neighbours__people_of_a_different_race",
    A124_03="question_a124_03_neighbours__heavy_drinkers",
    A124_04="question_a124_04_neighbours__emotionally_unstable_people",
    A124_05="question_a124_05_neighbours__muslims",
    A124_06="question_a124_06_neighbours__immigrants_foreign_workers",
    A124_07="question_a124_07_neighbours__people_who_have_aids",
    A124_08="question_a124_08_neighbours__drug_addicts",
    A124_09="question_a124_09_neighbours__homosexuals",
    A124_10="question_a124_10_neighbours__jews",
    A124_17="question_a124_17_neighbours__gypsies",
    A124_24="question_a124_24_neighbours__christians",
    A124_26="question_a124_26_neighbours__left_wing_extremists",
    A124_27="question_a124_27_neighbours__right_wing_extremists",
    A124_28="question_a124_28_neighbours__people_with_large_families",
    A124_29="question_a124_29_neighbours__hindus",
    A165="question_a165_most_people_can_be_trusted",
    A168="question_a168_do_you_think_most_people_try_to_take_advantage_of_you",
    A168A="question_a168a_do_you_think_most_people_try_to_take_advantage_of_you__10_point_scale",
    A169="question_a169_good_human_relationships",
    A170="question_a170_satisfaction_with_your_life",
    A173="question_a173_how_much_freedom_of_choice_and_control",
    B001="question_b001_would_give_part_of_my_income_for_the_environment__4_point_scale",
    B002="question_b002_increase_in_taxes_if_used_to_prevent_environmental_pollution",
    B003="question_b003_government_should_reduce_environmental_pollution",
    B005="question_b005_all_talk_about_the_environment_make_people_anxious",
    B006="question_b006_combatting_unemployment_we_have_to_accept_environmental_problems",
    B007="question_b007_protecting_environment_and_fighting_pollution_is_less_urgent_than_suggested",
    B008="question_b008_protecting_environment_vs_economic_growth",
    C001="question_c001_jobs_scarce__men_should_have_more_right_to_a_job_than_women__3_categories",
    C001_01="question_c001_01_jobs_scarce__men_should_have_more_right_to_a_job_than_women__5_point_scale",
    C002="question_c002_jobs_scarce__employers_should_give_priority_to__nation__people_than_immigrants__3_categories",
    C002_01="question_c002_01_jobs_scarce__employers_should_give_priority_to__nation__people_than_immigrants__5_point_scale",
    C004="question_c004_jobs_scarce__older_people_should_be_forced_to_retire",
    C005="question_c005_unfair_to_give_work_to_handicapped_people_when_able_bodied_people_can’t_find_jobs",
    C006="question_c006_satisfaction_with_financial_situation_of_household",
    C011="question_c011_important_in_a_job__good_pay",
    C012="question_c012_important_in_a_job__not_too_much_pressure",
    C013="question_c013_important_in_a_job__good_job_security",
    C014="question_c014_important_in_a_job__a_respected_job",
    C015="question_c015_important_in_a_job__good_hours",
    C016="question_c016_important_in_a_job__an_opportunity_to_use_initiative",
    C017="question_c017_important_in_a_job__generous_holidays",
    C018="question_c018_important_in_a_job__that_you_can_achieve_something",
    C019="question_c019_important_in_a_job__a_responsible_job",
    C020="question_c020_important_in_a_job__a_job_that_is_interesting",
    C021="question_c021_important_in_a_job__a_job_that_meets_one’s_abilities",
    C022="question_c022_important_in_a_job__pleasant_people_to_work_with",
    C023="question_c023_important_in_a_job__good_chances_for_promotion",
    C024="question_c024_important_in_a_job__a_useful_job_for_society",
    C025="question_c025_important_in_a_job__meeting_people",
    C027_90="question_c027_90_important_in_a_job__none_of_these",
    C029="question_c029_employed",
    C031="question_c031_degree_of_pride_in_your_work",
    C033="question_c033_job_satisfaction",
    C034="question_c034_freedom_decision_taking_in_job",
    C036="question_c036_to_develop_talents_you_need_to_have_a_job",
    C037="question_c037_humiliating_to_receive_money_without_having_to_work_for_it",
    C038="question_c038_people_who_don’t_work_turn_lazy",
    C039="question_c039_work_is_a_duty_towards_society",
    C040="question_c040_people_should_not_have_to_work_if_they_don’t_want_to",
    C041="question_c041_work_should_come_first_even_if_it_means_less_spare_time",
    C042B1="question_c042b1_why_people_work__work_is_like_a_business_transaction",
    C042B2="question_c042b2_why_people_work__i_do_the_best_i_can_regardless_of_pay",
    C042B3="question_c042b3_why_people_work__i_wouldn’t_work_if_i_didn’t_have_to",
    C042B4="question_c042b4_why_people_work__i_wouldn’t_work_if_work_interfered_my_life",
    C042B5="question_c042b5_why_people_work__work_most_important_in_my_life",
    C042B6="question_c042b6_why_people_work__i_never_had_a_paid_job",
    C042B7="question_c042b7_why_people_work__don’t_know",
    C059="question_c059_fairness__one_secretary_is_paid_more",
    C060="question_c060_how_business_and_industry_should_be_managed",
    C061="question_c061_following_instructions_at_work",
    D001="question_d001_how_much_do_you_trust_your_family__5_point_scale",
    D001_B="question_d001_b_how_much_do_you_trust_your_family__4_point_scale",
    D002="question_d002_satisfaction_with_home_life",
    D003="question_d003_sharing_with_partner__attitudes_towards_religion",
    D004="question_d004_sharing_with_partner__moral_standards",
    D005="question_d005_sharing_with_partner__social_attitudes",
    D006="question_d006_sharing_with_partner__political_views",
    D007="question_d007_sharing_with_partner__sexual_attitudes",
    D008="question_d008_sharing_with_partner__no_sharing_attitudes",
    D009="question_d009_sharing_with_partner__don’t_know_or_missing",
    D010="question_d010_sharing_with_parents__attitudes_towards_religion",
    D011="question_d011_sharing_with_parents__moral_standards",
    D012="question_d012_sharing_with_parents__social_attitudes",
    D013="question_d013_sharing_with_parents__political_views",
    D014="question_d014_sharing_with_parents__sexual_attitudes",
    D015="question_d015_sharing_with_parents__no_sharing_attitudes",
    D016="question_d016_sharing_with_parents__don’t_know_or_missing",
    D017="question_d017_ideal_number_of_children",
    D018="question_d018_child_needs_a_home_with_father_and_mother",
    D019="question_d019_a_woman_has_to_have_children_to_be_fulfilled",
    D020="question_d020_a_man_has_to_have_children_to_be_fulfilled",
    D022="question_d022_marriage_is_an_out_dated_institution",
    D023="question_d023_woman_as_a_single_parent",
    D024="question_d024_enjoy_sexual_freedom",
    D026="question_d026_long_term_relationship_is_necessary_to_be_happy",
    D026_03="question_d026_03_duty_towards_society_to_have_children",
    D026_05="question_d026_05_it_is_childs_duty_to_take_care_of_ill_parent",
    D027="question_d027_important_for_successful_marriage__faithfulness",
    D028="question_d028_important_for_successful_marriage__adequate_income",
    D029="question_d029_important_for_successful_marriage__same_social_background",
    D030="question_d030_important_for_successful_marriage__respect_and_appreciation",
    D031="question_d031_important_for_successful_marriage__religious_beliefs",
    D032="question_d032_important_for_successful_marriage__good_housing",
    D033="question_d033_important_for_successful_marriage__agreement_on_politics",
    D034="question_d034_important_for_successful_marriage__understanding_and_tolerance",
    D035="question_d035_important_for_successful_marriage__apart_from_in_laws",
    D036="question_d036_important_for_successful_marriage__happy_sexual_relationship",
    D037="question_d037_important_for_successful_marriage__sharing_household_chores",
    D038="question_d038_important_for_successful_marriage__children",
    D039="question_d039_important_for_successful_marriage__discussing_problems",
    D043="question_d043_important_for_successful_marriage__tastes_and_interests_in_common",
    D043_01="question_d043_01_important_for_successful_marriage__time_for_friends_and_personal_hobbies",
    D054="question_d054_one_of_main_goals_in_life_has_been_to_make_my_parents_proud",
    D056="question_d056_relationship_working_mother",
    D057="question_d057_being_a_housewife_just_as_fulfilling",
    D058="question_d058_husband_and_wife_should_both_contribute_to_income",
    D059="question_d059_men_make_better_political_leaders_than_women_do",
    D060="question_d060_university_is_more_important_for_a_boy_than_for_a_girl",
    D061="question_d061_pre_school_child_suffers_with_working_mother",
    D062="question_d062_women_want_a_home_and_children",
    D063="question_d063_job_best_way_for_women_to_be_independent__4_categories",
    D064="question_d064_fathers_are_well_suited_for_looking_after_children",
    D078="question_d078_men_make_better_business_executives_than_women_do",
    D081="question_d081_homosexual_couples_are_as_good_parents_as_other_couples",
    E001="question_e001_aims_of_country__first_choice",
    E002="question_e002_aims_of_country__second_choice",
    E003="question_e003_aims_of_respondent__first_choice",
    E004="question_e004_aims_of_respondent__second_choice",
    E005="question_e005_most_important__first_choice",
    E006="question_e006_most_important__second_choice",
    E012="question_e012_willingness_to_fight_for_country",
    E014="question_e014_future_changes__less_emphasis_on_money_and_material_possessions",
    E015="question_e015_future_changes__less_importance_placed_on_work",
    E016="question_e016_future_changes__more_emphasis_on_technology",
    E017="question_e017_future_changes__more_emphasis_on_individual",
    E018="question_e018_future_changes__greater_respect_for_authority",
    E019="question_e019_future_changes__more_emphasis_on_family_life",
    E020="question_e020_future_changes__a_simple_and_more_natural_lifestyle",
    E022="question_e022_opinion_about_scientific_advances",
    E023="question_e023_interest_in_politics",
    E025="question_e025_political_action__signing_a_petition",
    E026="question_e026_political_action__joining_in_boycotts",
    E027="question_e027_political_action__attending_lawful_peaceful_demonstrations",
    E028="question_e028_political_action__joining_unofficial_strikes",
    E029="question_e029_political_action__occupying_buildings_or_factories",
    E032="question_e032_freedom_or_equality",
    E033="question_e033_self_positioning_in_political_scale",
    E034="question_e034_basic_kinds_of_attitudes_concerning_society",
    E035="question_e035_income_equality",
    E036="question_e036_private_vs_state_ownership_of_business",
    E037="question_e037_government_responsibility",
    E038="question_e038_job_taking_of_the_unemployed",
    E039="question_e039_competition_good_or_harmful",
    E040="question_e040_hard_work_brings_success",
    E041="question_e041_wealth_accumulation",
    E042="question_e042_firms_and_freedom",
    E045="question_e045_major_changes_in_life",
    E046="question_e046_new_and_old_ideas",
    E047="question_e047_personal_characteristics__changes_worry_or_welcome_possibility",
    E048="question_e048_personal_characteristics__i_usually_count_on_being_successful_in_everything_i_do",
    E049="question_e049_personal_characteristics__i_enjoy_convincing_others_of_my_opinion",
    E050="question_e050_personal_characteristics__i_serve_as_a_model_for_others",
    E051="question_e051_personal_characteristics__i_am_good_at_getting_what_i_want",
    E052="question_e052_personal_characteristics__i_own_many_things_others_envy_me_for",
    E053="question_e053_personal_characteristics__i_like_to_assume_responsibility",
    E054="question_e054_personal_characteristics__i_am_rarely_unsure_about_how_i_should_behave",
    E055="question_e055_personal_characteristics__i_often_give_others_advice",
    E056="question_e056_personal_characteristics__none_of_the_above",
    E057="question_e057_the_economic_system_needs_fundamental_changes",
    E058="question_e058_our_government_should_be_made_much_more_open_to_the_public",
    E059="question_e059_allow_more_freedom_for_individuals",
    E060="question_e060_i_could_do_nothing_about_an_unjust_law",
    E061="question_e061_political_reform_is_moving_too_rapidly",
    E069_01="question_e069_01_confidence__churches",
    E069_02="question_e069_02_confidence__armed_forces",
    E069_03="question_e069_03_confidence__education_system",
    E069_04="question_e069_04_confidence__the_press",
    E069_05="question_e069_05_confidence__labour_unions",
    E069_06="question_e069_06_confidence__the_police",
    E069_07="question_e069_07_confidence__parliament",
    E069_08="question_e069_08_confidence__the_civil_services",
    E069_09="question_e069_09_confidence__social_security_system",
    E069_11="question_e069_11_confidence__the_government",
    E069_12="question_e069_12_confidence__the_political_parties",
    E069_13="question_e069_13_confidence__major_companies",
    E069_14="question_e069_14_confidence__the_environmental_protection_movement",
    E069_16="question_e069_16_confidence__health_care_system",
    E069_17="question_e069_17_confidence__justice_system_courts",
    E069_18="question_e069_18_confidence__the_european_union",
    E069_18A="question_e069_18a_confidence__major_regional_organization__combined_from_country_specific",
    E069_19="question_e069_19_confidence__nato",
    E069_20="question_e069_20_confidence__the_united_nations",
    E104="question_e104_approval__ecology_movement_or_nature_protection",
    E105="question_e105_approval__anti_nuclear_energy_movement",
    E106="question_e106_approval__disarmament_movement",
    E107="question_e107_approval__human_rights_movement",
    E108="question_e108_approval__women’s_movement",
    E109="question_e109_approval__anti_apartheid_movement",
    E110="question_e110_satisfaction_with_the_way_democracy_develops",
    E111="question_e111_rate_political_system_for_governing_country",
    E111_01="question_e111_01_satisfaction_with_the_political_system",
    E112="question_e112_rate_political_system_as_it_was_before",
    E114="question_e114_political_system__having_a_strong_leader",
    E115="question_e115_political_system__having_experts_make_decisions",
    E116="question_e116_political_system__having_the_army_rule",
    E117="question_e117_political_system__having_a_democratic_political_system",
    E118="question_e118_firm_party_leader_vs_cooperating_party_leader",
    E119="question_e119_government_order_vs_freedom",
    E120="question_e120_in_democracy_the_economic_system_runs_badly",
    E121="question_e121_democracies_are_indecisive_and_have_too_much_squabbling",
    E122="question_e122_democracies_aren’t_good_at_maintaining_order",
    E123="question_e123_democracy_may_have_problems_but_is_better",
    E124="question_e124_respect_for_individual_human_rights_nowadays",
    E125="question_e125_satisfaction_with_the_people_in_national_office",
    E129="question_e129_economic_aid_to_poorer_countries",
    E135="question_e135_who_should_decide__international_peacekeeping",
    E136="question_e136_who_should_decide__protection_of_the_environment",
    E137="question_e137_who_should_decide__aid_to_developing_countries",
    E138="question_e138_who_should_decide__refugees",
    E139="question_e139_who_should_decide__human_rights",
    E140="question_e140_country_cannot_solve_environmental_problems_by_itself",
    E141="question_e141_country_cannot_solve_crime_problems_by_itself",
    E142="question_e142_country_cannot_solve_employment_problems_by_itself",
    E143="question_e143_immigrant_policy",
    E144="question_e144_living_day_to_day_because_of_uncertain_future",
    E150="question_e150_how_often_follows_politics_in_the_news",
    E151="question_e151_give_authorities_information_to_help_justice",
    E152="question_e152_stick_to_own_affairs",
    E153="question_e153_feel_concerned_about_immediate_family",
    E154="question_e154_feel_concerned_about_people_in_the_neighbourhood",
    E155="question_e155_feel_concerned_about_people_in_the_region",
    E156="question_e156_feel_concerned_about_fellow_countrymen",
    E157="question_e157_feel_concerned_about_europeans",
    E158="question_e158_feel_concerned_about_human_kind",
    E159="question_e159_feel_concerned_about_elderly_people",
    E160="question_e160_feel_concerned_about_unemployed_people",
    E161="question_e161_feel_concerned_about_immigrants",
    E162="question_e162_feel_concerned_about_sick_and_disabled_people",
    E181C="question_e181c_which_political_party_would_you_vote_for_appeals_to_you__left_right_scale",
    E188="question_e188_frequency_watches_tv",
    E189="question_e189_tv_most_important_entertainment",
    E190="question_e190_why_are_there_people_living_in_need__first",
    E191="question_e191_why_are_there_people_living_in_need__second",
    E197="question_e197_opinion_on_terrorism",
    E224="question_e224_democracy__governments_tax_the_rich_and_subsidize_the_poor",
    E225="question_e225_democracy__religious_authorities_interpret_the_laws",
    E226="question_e226_democracy__people_choose_their_leaders_in_free_elections",
    E227="question_e227_democracy__people_receive_state_aid_for_unemployment",
    E228="question_e228_democracy__the_army_takes_over_when_government_is_incompetent",
    E229="question_e229_democracy__civil_rights_protect_people’s_liberty_against_oppression",
    E233="question_e233_democracy__women_have_the_same_rights_as_men",
    E233A="question_e233a_democracy__the_state_makes_people's_incomes_equal",
    E233B="question_e233b_democracy__people_obey_their_rulers",
    E235="question_e235_importance_of_democracy",
    E236="question_e236_democraticness_in_own_country",
    E263="question_e263_vote_in_elections__local_level",
    E264="question_e264_vote_in_elections__national_level",
    E265_01="question_e265_01_how_often_in_country’s_elections__votes_are_counted_fairly",
    E265_02="question_e265_02_how_often_in_country’s_elections__opposition_candidates_are_prevented_from_running",
    E265_03="question_e265_03_how_often_in_country’s_elections__tv_news_favors_the_governing_party",
    E265_04="question_e265_04_how_often_in_country’s_elections__voters_are_bribed",
    E265_05="question_e265_05_how_often_in_country’s_elections__journalists_provide_fair_coverage_of_elections",
    E265_06="question_e265_06_how_often_in_country’s_elections__election_officials_are_fair",
    E265_07="question_e265_07_how_often_in_country’s_election__rich_people_buy_elections",
    E265_08="question_e265_08_how_often_in_country’s_elections__voters_are_threatened_with_violence_at_the_polls",
    E290="question_e290_justifiable__political_violence",
    F001="question_f001_thinking_about_meaning_and_purpose_of_life",
    F003="question_f003_thinking_about_death",
    F004="question_f004_life_is_meaningful_because_god_exists",
    F005="question_f005_try_to_get_the_best_out_of_life",
    F006="question_f006_death_is_inevitable",
    F007="question_f007_death_has_meaning_if_you_believe_in_god",
    F008="question_f008_death_is_a_natural_resting_point",
    F009="question_f009_sorrow_has_meaning_if_you_believe_in_god",
    F010="question_f010_life_has_no_meaning",
    F022="question_f022_statement__good_and_evil",
    F024="question_f024_belong_to_religious_denomination",
    F025="question_f025_religious_denomination__major_groups",
    F026="question_f026_former_religious_denomination",
    F027="question_f027_which_former_religious_denomination",
    F028="question_f028_how_often_do_you_attend_religious_services",
    F029="question_f029_raised_religiously",
    F030="question_f030_attendance_religious_services_12_years_old",
    F031="question_f031_important__religious_service_birth",
    F032="question_f032_important__religious_service_marriage",
    F033="question_f033_important__religious_service_death",
    F034="question_f034_religious_person",
    F035="question_f035_churches_give_answers__moral_problems",
    F036="question_f036_churches_give_answers__the_problems_of_family_life",
    F037="question_f037_churches_give_answers__people’s_spiritual_needs",
    F038="question_f038_churches_give_answers__the_social_problems",
    F040="question_f040_churches_speak_out_on__disarmament",
    F041="question_f041_churches_speak_out_on__abortion",
    F042="question_f042_churches_speak_out_on__third_world_problems",
    F043="question_f043_churches_speak_out_on__extramarital_affairs",
    F044="question_f044_churches_speak_out_on__unemployment",
    F045="question_f045_churches_speak_out_on__racial_discrimination",
    F046="question_f046_churches_speak_out_on__euthanasia",
    F047="question_f047_churches_speak_out_on__homosexuality",
    F048="question_f048_churches_speak_out_on__ecology_and_environmental_issues",
    F049="question_f049_churches_speak_out_on__government_policy",
    F050="question_f050_believe_in__god",
    F051="question_f051_believe_in__life_after_death",
    F052="question_f052_believe_in__people_have_a_soul",
    F053="question_f053_believe_in__hell",
    F054="question_f054_believe_in__heaven",
    F055="question_f055_believe_in__sin",
    F057="question_f057_believe_in__re_incarnation",
    F059="question_f059_believe_in__devil",
    F060="question_f060_believe_in__resurrection_of_the_dead",
    F062="question_f062_personal_god_vs_spirit_or_life_force",
    F063="question_f063_how_important_is_god_in_your_life",
    F064="question_f064_get_comfort_and_strength_from_religion",
    F065="question_f065_moments_of_prayer_meditation",
    F066="question_f066_pray_to_god_outside_of_religious_services__i",
    F067="question_f067_pray_to_god_outside_of_religious_services__ii",
    F099="question_f099_lucky_charm_protects",
    F102="question_f102_politicians_who_don’t_believe_in_god_are_unfit_for_public_office",
    F103="question_f103_religious_leaders_should_not_influence_how_people_vote",
    F104="question_f104_better_if_more_people_with_strong_religious_beliefs_in_public_office",
    F105="question_f105_religious_leaders_should_not_influence_government",
    F114="question_f114_justifiable__claiming_government_benefits",
    F114A="question_f114a_justifiable__claiming_government_benefits_to_which_you_are_not_entitled",
    F115="question_f115_justifiable__avoiding_a_fare_on_public_transport",
    F116="question_f116_justifiable__cheating_on_taxes",
    F117="question_f117_justifiable__someone_accepting_a_bribe",
    F118="question_f118_justifiable__homosexuality",
    F119="question_f119_justifiable__prostitution",
    F120="question_f120_justifiable__abortion",
    F121="question_f121_justifiable__divorce",
    F122="question_f122_justifiable__euthanasia",
    F123="question_f123_justifiable__suicide",
    F125="question_f125_justifiable__joyriding",
    F126="question_f126_justifiable__taking_soft_drugs",
    F127="question_f127_justifiable__lying",
    F128="question_f128_justifiable__adultery",
    F129="question_f129_justifiable__throwing_away_litter",
    F130="question_f130_justifiable__driving_under_influence_of_alcohol",
    F131="question_f131_justifiable__paying_cash",
    F132="question_f132_justifiable__having_casual_sex",
    F135="question_f135_justifiable__sex_under_the_legal_age_of_consent",
    F136="question_f136_justifiable__political_assassination",
    F137="question_f137_justifiable__experiments_with_human_embryos",
    F138="question_f138_justifiable__manipulation_of_food",
    F139="question_f139_justifiable__buy_stolen_goods",
    F140="question_f140_justifiable__keeping_money_that_you_have_found",
    F141="question_f141_justifiable__fighting_with_the_police",
    F142="question_f142_justifiable__failing_to_report_damage_you’ve_done_accidentally_to_a_parked_vehicle",
    F143="question_f143_justifiable__threatening_workers_who_refuse_to_join_a_strike",
    F144="question_f144_justifiable__killing_in_self_defence",
    F144_01="question_f144_01_justifiable__invitro_fertilization",
    F144_02="question_f144_02_justifiable__death_penalty",
    G001="question_g001_geographical_groups_belonging_to_first",
    G002="question_g002_geographical_groups_belonging_to_second",
    G005="question_g005_citizen_of__country",
    G006="question_g006_how_proud_of_nationality",
    G007_01="question_g007_01_trust__other_people_in_country",
    G007_18_B="question_g007_18_b_trust__your_neighborhood__b",
    G007_33_B="question_g007_33_b_trust__people_you_know_personally__b",
    G007_34_B="question_g007_34_b_trust__people_you_meet_for_the_first_time__b",
    G007_35_B="question_g007_35_b_trust__people_of_another_religion__b",
    G007_36_B="question_g007_36_b_trust__people_of_another_nationality__b",
    G014="question_g014_opinion_european_union",
    G027A="question_g027a_respondent_immigrant",
    G033="question_g033_important__to_have_been_born_in__country",
    G034="question_g034_important__to_respect__country_nationality__political_institutions_and_laws",
    G035="question_g035_important__to_have__country_nationality__ancestry",
    G036="question_g036_important__to_be_able_to_speak__country_language",
    G038="question_g038_immigrants_take_away_jobs_from__nationality",
    G040="question_g040_immigrants_increase_crime_problems",
    G041="question_g041_immigrants_are_a_strain_on_welfare_system",
    G043="question_g043_immigrants_maintain_own_take_over_customs",
    G051="question_g051_european_union_enlargement",
    G052="question_g052_evaluate_the_impact_of_immigrants_on_the_development_of__your_country",
    G062="question_g062_how_close_you_feel__continent__e_g_europe_asia_etc",
    G063="question_g063_how_close_you_feel__world",
    G255="question_g255_how_close_you_feel__your__village_town_or_city",
    G256="question_g256_how_close_do_you_feel__to_your_county_region_district",
    G257="question_g257_how_close_do_you_feel__to_country",
    H009="question_h009_government_has_the_right__keep_people_under_video_surveillance_in_public_areas",
    H010="question_h010_government_has_the_right__monitor_all_e_mails_and_any_other_information_exchanged_on_the_internet",
    H011="question_h011_government_has_the_right__collect_information_about_anyone_living_in__country__without_their_knowledge",
    V011="question_v011_mother_liked_to_read_books",
    V012="question_v012_discussed_politcs_with_mother",
    V013="question_v013_mother_liked_to_follow_the_news",
    V014="question_v014_parents_had_problems_making_ends_meet",
    V015="question_v015_father_liked_to_read_books",
    V016="question_v016_discussed_politcs_with_father",
    V017="question_v017_father_liked_to_follow_the_news",
    V018="question_v018_parents_had_problems_replacing_broken_things",
)

EVS_WVS_ANSWER_COLUMNS = dict(
    A001="question_a001_important_in_life__family",
    A002="question_a002_important_in_life__friends",
    A003="question_a003_important_in_life__leisure_time",
    A004="question_a004_important_in_life__politics",
    A005="question_a005_important_in_life__work",
    A006="question_a006_important_in_life__religion",
    A008="question_a008_feeling_of_happiness",
    A009="question_a009_state_of_health__subjective",
    A027="question_a027_important_child_qualities__good_manners",
    A029="question_a029_important_child_qualities__independence",
    A030="question_a030_important_child_qualities__hard_work",
    A032="question_a032_important_child_qualities__feeling_of_responsibility",
    A034="question_a034_important_child_qualities__imagination",
    A035="question_a035_important_child_qualities__tolerance_and_respect_for_other_people",
    A038="question_a038_important_child_qualities__thrift_saving_money_and_things",
    A039="question_a039_important_child_qualities__determination_perseverance",
    A040="question_a040_important_child_qualities__religious_faith",
    A041="question_a041_important_child_qualities__unselfishness",
    A042="question_a042_important_child_qualities__obedience",
    A065="question_a065_member__belong_to_religious_organization",
    A066="question_a066_member__belong_to_education_arts_music_or_cultural_activities",
    A067="question_a067_member__belong_to_labour_unions",
    A068="question_a068_member__belong_to_political_parties",
    A071="question_a071_member__belong_to_conservation_the_environment_ecology_animal_rights",
    A072="question_a072_member__belong_to_professional_associations",
    A074="question_a074_member__belong_to_sports_or_recreation",
    A078="question_a078_member__belong_to_consumer_groups",
    A079="question_a079_member__belong_to_other_groups",
    A080_01="question_a080_01_member__belong_to_humanitarian_or_charitable_organization",
    A080_02="question_a080_02_member__belong_to_self_help_group_mutual_aid_group",
    A124_02="question_a124_02_neighbours__people_of_a_different_race",
    A124_03="question_a124_03_neighbours__heavy_drinkers",
    A124_06="question_a124_06_neighbours__immigrants_foreign_workers",
    A124_08="question_a124_08_neighbours__drug_addicts",
    A124_09="question_a124_09_neighbours__homosexuals",
    A165="question_a165_most_people_can_be_trusted",
    A170="question_a170_satisfaction_with_your_life",
    A173="question_a173_how_much_freedom_of_choice_and_control",
    B008="question_b008_protecting_environment_vs_economic_growth",
    C001="question_c001_jobs_scarce__men_should_have_more_right_to_a_job_than_women__3_categories",
    C001_01="question_c001_01_jobs_scarce__men_should_have_more_right_to_a_job_than_women__5_point_scale",
    C002="question_c002_jobs_scarce__employers_should_give_priority_to__nation__people_than_immigrants__3_categories",
    C002_01="question_c002_01_jobs_scarce__employers_should_give_priority_to__nation__people_than_immigrants__5_point_scale",
    C038="question_c038_people_who_don’t_work_turn_lazy",
    C039="question_c039_work_is_a_duty_towards_society",
    C041="question_c041_work_should_come_first_even_if_it_means_less_spare_time",
    D001_B="question_d001_b_how_much_do_you_trust_your_family__4_point_scale",
    D026_03="question_d026_03_duty_towards_society_to_have_children",
    D026_05="question_d026_05_it_is_childs_duty_to_take_care_of_ill_parent",
    D054="question_d054_one_of_main_goals_in_life_has_been_to_make_my_parents_proud",
    D059="question_d059_men_make_better_political_leaders_than_women_do",
    D060="question_d060_university_is_more_important_for_a_boy_than_for_a_girl",
    D061="question_d061_pre_school_child_suffers_with_working_mother",
    D078="question_d078_men_make_better_business_executives_than_women_do",
    D081="question_d081_homosexual_couples_are_as_good_parents_as_other_couples",
    E001="question_e001_aims_of_country__first_choice",
    E002="question_e002_aims_of_country__second_choice",
    E003="question_e003_aims_of_respondent__first_choice",
    E004="question_e004_aims_of_respondent__second_choice",
    E012="question_e012_willingness_to_fight_for_country",
    E015="question_e015_future_changes__less_importance_placed_on_work",
    E018="question_e018_future_changes__greater_respect_for_authority",
    E023="question_e023_interest_in_politics",
    E025="question_e025_political_action__signing_a_petition",
    E026="question_e026_political_action__joining_in_boycotts",
    E027="question_e027_political_action__attending_lawful_peaceful_demonstrations",
    E028="question_e028_political_action__joining_unofficial_strikes",
    E033="question_e033_self_positioning_in_political_scale",
    E035="question_e035_income_equality",
    E036="question_e036_private_vs_state_ownership_of_business",
    E037="question_e037_government_responsibility",
    E039="question_e039_competition_good_or_harmful",
    E069_01="question_e069_01_confidence__churches",
    E069_02="question_e069_02_confidence__armed_forces",
    E069_04="question_e069_04_confidence__the_press",
    E069_05="question_e069_05_confidence__labour_unions",
    E069_06="question_e069_06_confidence__the_police",
    E069_07="question_e069_07_confidence__parliament",
    E069_08="question_e069_08_confidence__the_civil_services",
    E069_11="question_e069_11_confidence__the_government",
    E069_12="question_e069_12_confidence__the_political_parties",
    E069_13="question_e069_13_confidence__major_companies",
    E069_14="question_e069_14_confidence__the_environmental_protection_movement",
    E069_17="question_e069_17_confidence__justice_system_courts",
    E069_18="question_e069_18_confidence__the_european_union",
    E069_18A="question_e069_18a_confidence__major_regional_organization__combined_from_country_specific",
    E069_20="question_e069_20_confidence__the_united_nations",
    E111_01="question_e111_01_satisfaction_with_the_political_system",
    E114="question_e114_political_system__having_a_strong_leader",
    E115="question_e115_political_system__having_experts_make_decisions",
    E116="question_e116_political_system__having_the_army_rule",
    E117="question_e117_political_system__having_a_democratic_political_system",
    E224="question_e224_democracy__governments_tax_the_rich_and_subsidize_the_poor",
    E225="question_e225_democracy__religious_authorities_interpret_the_laws",
    E226="question_e226_democracy__people_choose_their_leaders_in_free_elections",
    E227="question_e227_democracy__people_receive_state_aid_for_unemployment",
    E228="question_e228_democracy__the_army_takes_over_when_government_is_incompetent",
    E229="question_e229_democracy__civil_rights_protect_people’s_liberty_against_oppression",
    E233="question_e233_democracy__women_have_the_same_rights_as_men",
    E233A="question_e233a_democracy__the_state_makes_people's_incomes_equal",
    E233B="question_e233b_democracy__people_obey_their_rulers",
    E235="question_e235_importance_of_democracy",
    E236="question_e236_democraticness_in_own_country",
    E263="question_e263_vote_in_elections__local_level",
    E264="question_e264_vote_in_elections__national_level",
    E265_01="question_e265_01_how_often_in_country’s_elections__votes_are_counted_fairly",
    E265_02="question_e265_02_how_often_in_country’s_elections__opposition_candidates_are_prevented_from_running",
    E265_03="question_e265_03_how_often_in_country’s_elections__tv_news_favors_the_governing_party",
    E265_04="question_e265_04_how_often_in_country’s_elections__voters_are_bribed",
    E265_05="question_e265_05_how_often_in_country’s_elections__journalists_provide_fair_coverage_of_elections",
    E265_06="question_e265_06_how_often_in_country’s_elections__election_officials_are_fair",
    E265_07="question_e265_07_how_often_in_country’s_election__rich_people_buy_elections",
    E265_08="question_e265_08_how_often_in_country’s_elections__voters_are_threatened_with_violence_at_the_polls",
    E290="question_e290_justifiable__political_violence",
    F025="question_f025_religious_denomination__major_groups",
    F028="question_f028_how_often_do_you_attend_religious_services",
    F034="question_f034_religious_person",
    F050="question_f050_believe_in__god",
    F051="question_f051_believe_in__life_after_death",
    F053="question_f053_believe_in__hell",
    F054="question_f054_believe_in__heaven",
    F063="question_f063_how_important_is_god_in_your_life",
    F114A="question_f114a_justifiable__claiming_government_benefits_to_which_you_are_not_entitled",
    F115="question_f115_justifiable__avoiding_a_fare_on_public_transport",
    F116="question_f116_justifiable__cheating_on_taxes",
    F117="question_f117_justifiable__someone_accepting_a_bribe",
    F118="question_f118_justifiable__homosexuality",
    F119="question_f119_justifiable__prostitution",
    F120="question_f120_justifiable__abortion",
    F121="question_f121_justifiable__divorce",
    F122="question_f122_justifiable__euthanasia",
    F123="question_f123_justifiable__suicide",
    F132="question_f132_justifiable__having_casual_sex",
    F144_02="question_f144_02_justifiable__death_penalty",
    G005="question_g005_citizen_of__country",
    G006="question_g006_how_proud_of_nationality",
    G007_18_B="question_g007_18_b_trust__your_neighborhood__b",
    G007_33_B="question_g007_33_b_trust__people_you_know_personally__b",
    G007_34_B="question_g007_34_b_trust__people_you_meet_for_the_first_time__b",
    G007_35_B="question_g007_35_b_trust__people_of_another_religion__b",
    G007_36_B="question_g007_36_b_trust__people_of_another_nationality__b",
    G027A="question_g027a_respondent_immigrant",
    G052="question_g052_evaluate_the_impact_of_immigrants_on_the_development_of__your_country",
    G062="question_g062_how_close_you_feel__continent__e_g_europe_asia_etc",
    G063="question_g063_how_close_you_feel__world",
    G255="question_g255_how_close_you_feel__your__village_town_or_city",
    G256="question_g256_how_close_do_you_feel__to_your_county_region_district",
    G257="question_g257_how_close_do_you_feel__to_country",
    H009="question_h009_government_has_the_right__keep_people_under_video_surveillance_in_public_areas",
    H010="question_h010_government_has_the_right__monitor_all_e_mails_and_any_other_information_exchanged_on_the_internet",
    H011="question_h011_government_has_the_right__collect_information_about_anyone_living_in__country__without_their_knowledge",
)

CATEGORICAL_COLUMNS = [
    "A025",
    "A026",
    "A169",
    "B008",
    "C001",
    "C002",
    "C004",
    "C005",
    "C060",
    "C061",
    "D023",
    "D024",
    "E001",
    "E002",
    "E003",
    "E004",
    "E005",
    "E006",
    "E022",
    "E032",
    "E034",
    "E118",
    "E119",
    "E135",
    "E136",
    "E137",
    "E138",
    "E139",
    "E181C",
    "E190",
    "E191",
    "E197",
    "E263",
    "E264",
    "F004",
    "F005",
    "F006",
    "F007",
    "F008",
    "F009",
    "F010",
    "F022",
    "F025",
    "F027",
    "F062",
]
