# Dieses Script liest eine CSV-Datendatei in GNU R ein.
# Beim Einlesen werden für alle Variablen Beschriftungen (comment) angelegt.
# Die Beschriftungen für Werte wird ebenfalls als Attribute (attr) abgelegt.

ds_file = file.choose()
# setwd("./")
# ds_file = "rdata_finch_2024-09-23_16-45.csv"

options(encoding = "UTF-8")
ds = read.delim(
  file=ds_file, encoding="UTF-8", fileEncoding="UTF-8",
  header = FALSE, sep = "\t", quote = "\"",
  dec = ".", row.names = NULL,
  col.names = c(
    "CASE","SERIAL","REF","QUESTNNR","MODE","STARTED","AN03","DI03","DI03_01",
    "DI03_03","DI03_04","PI01_01","PI01_01a","PI02","PI03_01","PI03_02","PI03_03",
    "PI03_04","QU01_01","QU01_02","QU01_03","QU01_04","QU01_05","QU01_06","QU01_07",
    "QU01_08","QU01_09","QU01_10","QU03_01","QU04_01","QU04_02","QU04_03","QU04_04",
    "QU04_05","QU04_06","QU04_07","QU05_01","QU05_02","QU05_03","QU05_04","QU06_05",
    "QU06_04","QU06_06","QU06_07","TIME001","TIME002","TIME003","TIME004","TIME005",
    "TIME006","TIME007","TIME008","TIME009","TIME_SUM","MAILSENT","LASTDATA",
    "FINISHED","Q_VIEWER","LASTPAGE","MAXPAGE","MISSING","MISSREL","TIME_RSI"
  ),
  as.is = TRUE,
  colClasses = c(
    CASE="numeric", SERIAL="character", REF="character", QUESTNNR="character",
    MODE="factor", STARTED="POSIXct", AN03="numeric", DI03="numeric",
    DI03_01="logical", DI03_03="logical", DI03_04="logical", PI01_01="numeric",
    PI01_01a="logical", PI02="numeric", PI03_01="numeric", PI03_02="numeric",
    PI03_03="numeric", PI03_04="numeric", QU01_01="numeric", QU01_02="numeric",
    QU01_03="numeric", QU01_04="numeric", QU01_05="numeric", QU01_06="numeric",
    QU01_07="numeric", QU01_08="numeric", QU01_09="numeric", QU01_10="numeric",
    QU03_01="character", QU04_01="numeric", QU04_02="numeric",
    QU04_03="numeric", QU04_04="numeric", QU04_05="numeric", QU04_06="numeric",
    QU04_07="numeric", QU05_01="numeric", QU05_02="numeric", QU05_03="numeric",
    QU05_04="numeric", QU06_05="numeric", QU06_04="numeric", QU06_06="numeric",
    QU06_07="numeric", TIME001="integer", TIME002="integer", TIME003="integer",
    TIME004="integer", TIME005="integer", TIME006="integer", TIME007="integer",
    TIME008="integer", TIME009="integer", TIME_SUM="integer",
    MAILSENT="POSIXct", LASTDATA="POSIXct", FINISHED="logical",
    Q_VIEWER="logical", LASTPAGE="numeric", MAXPAGE="numeric",
    MISSING="numeric", MISSREL="numeric", TIME_RSI="numeric"
  ),
  skip = 1,
  check.names = TRUE, fill = TRUE,
  strip.white = FALSE, blank.lines.skip = TRUE,
  comment.char = "",
  na.strings = ""
)

row.names(ds) = ds$CASE

rm(ds_file)

attr(ds, "project") = "finch"
attr(ds, "description") = "Finch"
attr(ds, "date") = "2024-09-23 16:45:28"
attr(ds, "server") = "https://befragungen.ovgu.de"

# Variable und Value Labels
ds$AN03 = factor(ds$AN03, levels=c("1","2","-9"), labels=c("Accept","Decline","[NA] nicht beantwortet"), ordered=FALSE)
ds$PI02 = factor(ds$PI02, levels=c("1","2","3","4","-9"), labels=c("female","male","diverse","prefer not to say","[NA] nicht beantwortet"), ordered=FALSE)
attr(ds$DI03_01,"F") = "nicht gewählt"
attr(ds$DI03_01,"T") = "ausgewählt"
attr(ds$DI03_03,"F") = "nicht gewählt"
attr(ds$DI03_03,"T") = "ausgewählt"
attr(ds$DI03_04,"F") = "nicht gewählt"
attr(ds$DI03_04,"T") = "ausgewählt"
attr(ds$PI01_01a,"F") = "nicht gewählt"
attr(ds$PI01_01a,"T") = "ausgewählt"
attr(ds$PI03_01,"1") = "very unfamiliar"
attr(ds$PI03_01,"5") = "very familiar"
attr(ds$PI03_02,"1") = "very unfamiliar"
attr(ds$PI03_02,"5") = "very familiar"
attr(ds$PI03_03,"1") = "very unfamiliar"
attr(ds$PI03_03,"5") = "very familiar"
attr(ds$PI03_04,"1") = "very unfamiliar"
attr(ds$PI03_04,"5") = "very familiar"
attr(ds$QU01_01,"1") = "Strongly disagree"
attr(ds$QU01_01,"5") = "Strongly agree"
attr(ds$QU01_02,"1") = "Strongly agree"
attr(ds$QU01_02,"5") = "Strongly disagree"
attr(ds$QU01_03,"1") = "Strongly disagree"
attr(ds$QU01_03,"5") = "Strongly agree"
attr(ds$QU01_04,"1") = "Strongly agree"
attr(ds$QU01_04,"5") = "Strongly disagree"
attr(ds$QU01_05,"1") = "Strongly disagree"
attr(ds$QU01_05,"5") = "Strongly agree"
attr(ds$QU01_06,"1") = "Strongly agree"
attr(ds$QU01_06,"5") = "Strongly disagree"
attr(ds$QU01_07,"1") = "Strongly disagree"
attr(ds$QU01_07,"5") = "Strongly agree"
attr(ds$QU01_08,"1") = "Strongly agree"
attr(ds$QU01_08,"5") = "Strongly disagree"
attr(ds$QU01_09,"1") = "Strongly disagree"
attr(ds$QU01_09,"5") = "Strongly agree"
attr(ds$QU01_10,"1") = "Strongly agree"
attr(ds$QU01_10,"5") = "Strongly disagree"
attr(ds$QU04_01,"1") = "Strongly disagree"
attr(ds$QU04_01,"5") = "Strongly agree"
attr(ds$QU04_02,"1") = "Strongly disagree"
attr(ds$QU04_02,"5") = "Strongly agree"
attr(ds$QU04_03,"1") = "Strongly disagree"
attr(ds$QU04_03,"5") = "Strongly agree"
attr(ds$QU04_04,"1") = "Strongly disagree"
attr(ds$QU04_04,"5") = "Strongly agree"
attr(ds$QU04_05,"1") = "Strongly disagree"
attr(ds$QU04_05,"5") = "Strongly agree"
attr(ds$QU04_06,"1") = "Strongly disagree"
attr(ds$QU04_06,"5") = "Strongly agree"
attr(ds$QU04_07,"1") = "Strongly disagree"
attr(ds$QU04_07,"5") = "Strongly agree"
attr(ds$QU05_01,"1") = "Strongly disagree"
attr(ds$QU05_01,"5") = "Strongly agree"
attr(ds$QU05_02,"1") = "Strongly disagree"
attr(ds$QU05_02,"5") = "Strongly agree"
attr(ds$QU05_03,"1") = "Strongly disagree"
attr(ds$QU05_03,"5") = "Strongly agree"
attr(ds$QU05_04,"1") = "Strongly disagree"
attr(ds$QU05_04,"5") = "Strongly agree"
attr(ds$QU06_05,"1") = "Strongly disagree"
attr(ds$QU06_05,"5") = "Strongly agree"
attr(ds$QU06_04,"1") = "Strongly disagree"
attr(ds$QU06_04,"5") = "Strongly agree"
attr(ds$QU06_06,"1") = "Strongly disagree"
attr(ds$QU06_06,"5") = "Strongly agree"
attr(ds$QU06_07,"1") = "Strongly disagree"
attr(ds$QU06_07,"5") = "Strongly agree"
attr(ds$FINISHED,"F") = "abgebrochen"
attr(ds$FINISHED,"T") = "ausgefüllt"
attr(ds$Q_VIEWER,"F") = "Teilnehmer"
attr(ds$Q_VIEWER,"T") = "Durchklicker"
comment(ds$SERIAL) = "Personenkennung oder Teilnahmecode (sofern verwendet)"
comment(ds$REF) = "Referenz (sofern im Link angegeben)"
comment(ds$QUESTNNR) = "Fragebogen, der im Interview verwendet wurde"
comment(ds$MODE) = "Interview-Modus"
comment(ds$STARTED) = "Zeitpunkt zu dem das Interview begonnen hat (Europe/Berlin)"
comment(ds$AN03) = "ConsentQuestion"
comment(ds$DI03) = "Questions-Q: Ausweichoption (negativ) oder Anzahl ausgewählter Optionen"
comment(ds$DI03_01) = "Questions-Q: Do any of the main factors interact with each other?"
comment(ds$DI03_03) = "Questions-Q: Look at the general diabetes risk per age. How does the persons drug use (alcohol and smoking) impact this curve?"
comment(ds$DI03_04) = "Questions-Q: If the person starts drinking today, how will this change their prediction?"
comment(ds$PI01_01) = "Age:  ... years old"
comment(ds$PI01_01a) = "Age:  ... years old: prefer not to say"
comment(ds$PI02) = "Gender"
comment(ds$PI03_01) = "Experience: Machine Learning"
comment(ds$PI03_02) = "Experience: Explainable Artificial Intelligence (xAI)"
comment(ds$PI03_03) = "Experience: Partial Dependence Plots (PDP)"
comment(ds$PI03_04) = "Experience: SHAP plots"
comment(ds$QU01_01) = "SUS: I think that I would like to use this system frequently."
comment(ds$QU01_02) = "SUS: I found the system unnecessarily complex. (umgepolt)"
comment(ds$QU01_03) = "SUS: I thought the system was easy to use."
comment(ds$QU01_04) = "SUS: I think that I would need the support of a technical person to be able to use this system. (umgepolt)"
comment(ds$QU01_05) = "SUS: I found the various functions in this system were well integrated."
comment(ds$QU01_06) = "SUS: I thought there was too much inconsistency in this system. (umgepolt)"
comment(ds$QU01_07) = "SUS: I would imagine that most people would learn to use this system very quickly	."
comment(ds$QU01_08) = "SUS: I found the system very cumbersome to use. (umgepolt)"
comment(ds$QU01_09) = "SUS: I felt very confident using the system."
comment(ds$QU01_10) = "SUS: I needed to learn a lot of things before I could get going with this system (umgepolt)"
comment(ds$QU03_01) = "Comments: [01]"
comment(ds$QU04_01) = "TXAI: I am confident in this tool. I feel that it works well."
comment(ds$QU04_02) = "TXAI: The outputs of this tool are very predictable."
comment(ds$QU04_03) = "TXAI: This tool is very reliable. I can count on it to be correct all the time."
comment(ds$QU04_04) = "TXAI: I feel safe that when I rely on this tool I will get the right answers."
comment(ds$QU04_05) = "TXAI: This tool is efficient in that it works very quickly."
comment(ds$QU04_06) = "TXAI: This tool can perform the task better than a novice human user."
comment(ds$QU04_07) = "TXAI: I like using this tool for decision making."
comment(ds$QU05_01) = "Custom: The tool emphasizes what is special or unique in the data."
comment(ds$QU05_02) = "Custom: The tool distinguishes positive and negative contributions of features."
comment(ds$QU05_03) = "Custom: The tool offers an easily interpretable summary of the features."
comment(ds$QU05_04) = "Custom: The tool helps me validate if I can trust the shown results."
comment(ds$QU06_05) = "Custom_Vis: The charts provided by the tool are visually pleasing."
comment(ds$QU06_04) = "Custom_Vis: The charts provided by the tool are easy to interpret."
comment(ds$QU06_06) = "Custom_Vis: The charts provided by the tool are helpful for finding feature interactions."
comment(ds$QU06_07) = "Custom_Vis: The charts provided by the tool are helpful for interpreting feature interactions."
comment(ds$TIME001) = "Verweildauer Seite 1"
comment(ds$TIME002) = "Verweildauer Seite 2"
comment(ds$TIME003) = "Verweildauer Seite 3"
comment(ds$TIME004) = "Verweildauer Seite 4"
comment(ds$TIME005) = "Verweildauer Seite 5"
comment(ds$TIME006) = "Verweildauer Seite 6"
comment(ds$TIME007) = "Verweildauer Seite 7"
comment(ds$TIME008) = "Verweildauer Seite 8"
comment(ds$TIME009) = "Verweildauer Seite 9"
comment(ds$TIME_SUM) = "Verweildauer gesamt (ohne Ausreißer)"
comment(ds$MAILSENT) = "Versandzeitpunkt der Einladungsmail (nur für nicht-anonyme Adressaten)"
comment(ds$LASTDATA) = "Zeitpunkt als der Datensatz das letzte mal geändert wurde"
comment(ds$FINISHED) = "Wurde die Befragung abgeschlossen (letzte Seite erreicht)?"
comment(ds$Q_VIEWER) = "Hat der Teilnehmer den Fragebogen nur angesehen, ohne die Pflichtfragen zu beantworten?"
comment(ds$LASTPAGE) = "Seite, die der Teilnehmer zuletzt bearbeitet hat"
comment(ds$MAXPAGE) = "Letzte Seite, die im Fragebogen bearbeitet wurde"
comment(ds$MISSING) = "Anteil fehlender Antworten in Prozent"
comment(ds$MISSREL) = "Anteil fehlender Antworten (gewichtet nach Relevanz)"
comment(ds$TIME_RSI) = "Ausfüll-Geschwindigkeit (relativ)"



# Assure that the comments are retained in subsets
as.data.frame.avector = as.data.frame.vector
`[.avector` <- function(x,i,...) {
  r <- NextMethod("[")
  mostattributes(r) <- attributes(x)
  r
}
ds_tmp = data.frame(
  lapply(ds, function(x) {
    structure( x, class = c("avector", class(x) ) )
  } )
)
mostattributes(ds_tmp) = attributes(ds)
ds = ds_tmp
rm(ds_tmp)

