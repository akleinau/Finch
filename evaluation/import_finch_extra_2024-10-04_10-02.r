# Dieses Script liest eine CSV-Datendatei in GNU R ein.
# Beim Einlesen werden für alle Variablen Beschriftungen (comment) angelegt.
# Die Beschriftungen für Werte wird ebenfalls als Attribute (attr) abgelegt.

ds_file = file.choose()
# setwd("./")
# ds_file = "rdata_finch_extra_2024-10-04_10-02.csv"

options(encoding = "UTF-8")
ds = read.delim(
  file=ds_file, encoding="UTF-8", fileEncoding="UTF-8",
  header = FALSE, sep = "\t", quote = "\"",
  dec = ".", row.names = NULL,
  col.names = c(
    "CASE","SERIAL","REF","QUESTNNR","MODE","STARTED","QE01_01","QE01_02","QE01_03",
    "QE01_04","QE01_05","QE01_06","QE01_07","QE01_08","QE01_09","TIME001",
    "TIME_SUM","MAILSENT","LASTDATA","FINISHED","Q_VIEWER","LASTPAGE","MAXPAGE",
    "MISSING","MISSREL","TIME_RSI"
  ),
  as.is = TRUE,
  colClasses = c(
    CASE="numeric", SERIAL="character", REF="character", QUESTNNR="character",
    MODE="factor", STARTED="POSIXct", QE01_01="numeric", QE01_02="numeric",
    QE01_03="numeric", QE01_04="numeric", QE01_05="numeric", QE01_06="numeric",
    QE01_07="numeric", QE01_08="numeric", QE01_09="numeric", TIME001="integer",
    TIME_SUM="integer", MAILSENT="POSIXct", LASTDATA="POSIXct",
    FINISHED="logical", Q_VIEWER="logical", LASTPAGE="numeric",
    MAXPAGE="numeric", MISSING="numeric", MISSREL="numeric", TIME_RSI="numeric"
  ),
  skip = 1,
  check.names = TRUE, fill = TRUE,
  strip.white = FALSE, blank.lines.skip = TRUE,
  comment.char = "",
  na.strings = ""
)

row.names(ds) = ds$CASE

rm(ds_file)

attr(ds, "project") = "finch_extra"
attr(ds, "description") = "Finch Extra"
attr(ds, "date") = "2024-10-04 10:02:08"
attr(ds, "server") = "https://befragungen.ovgu.de"

# Variable und Value Labels
attr(ds$QE01_01,"1") = "I completely disagree"
attr(ds$QE01_01,"5") = "I completely agree"
attr(ds$QE01_02,"1") = "I completely disagree"
attr(ds$QE01_02,"5") = "I completely agree"
attr(ds$QE01_03,"1") = "I completely disagree"
attr(ds$QE01_03,"5") = "I completely agree"
attr(ds$QE01_04,"1") = "I completely disagree"
attr(ds$QE01_04,"5") = "I completely agree"
attr(ds$QE01_05,"1") = "I completely disagree"
attr(ds$QE01_05,"5") = "I completely agree"
attr(ds$QE01_06,"1") = "I completely disagree"
attr(ds$QE01_06,"5") = "I completely agree"
attr(ds$QE01_07,"1") = "I completely disagree"
attr(ds$QE01_07,"5") = "I completely agree"
attr(ds$QE01_08,"1") = "I completely disagree"
attr(ds$QE01_08,"5") = "I completely agree"
attr(ds$QE01_09,"1") = "I completely disagree"
attr(ds$QE01_09,"5") = "I completely agree"
attr(ds$FINISHED,"F") = "abgebrochen"
attr(ds$FINISHED,"T") = "ausgefüllt"
attr(ds$Q_VIEWER,"F") = "Teilnehmer"
attr(ds$Q_VIEWER,"T") = "Durchklicker"
comment(ds$SERIAL) = "Personenkennung oder Teilnahmecode (sofern verwendet)"
comment(ds$REF) = "Referenz (sofern im Link angegeben)"
comment(ds$QUESTNNR) = "Fragebogen, der im Interview verwendet wurde"
comment(ds$MODE) = "Interview-Modus"
comment(ds$STARTED) = "Zeitpunkt zu dem das Interview begonnen hat (Europe/Berlin)"
comment(ds$QE01_01) = "ESG: From the explanation, I understand how the model works."
comment(ds$QE01_02) = "ESG: This explanation of how the model works is satisfying."
comment(ds$QE01_03) = "ESG: This explanation of how the model works has sufficient detail."
comment(ds$QE01_04) = "ESG: This explanation seems complete."
comment(ds$QE01_05) = "ESG: This explanation shows me how accurate the model is."
comment(ds$QE01_06) = "ESG: This explanation shows me how reliable the model is."
comment(ds$QE01_07) = "ESG: This explanation tells me how to use the model."
comment(ds$QE01_08) = "ESG: This explanation is useful to my goals."
comment(ds$QE01_09) = "ESG: This explanation helps me know when I should trust and not trust the model."
comment(ds$TIME001) = "Verweildauer Seite 1"
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
ds_extra = ds_tmp
rm(ds_tmp)

