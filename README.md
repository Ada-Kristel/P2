# P2
Main files for reconstruction

Datasættet til at køre filerne findes på linket:
http://mridata.org/list?csrfmiddlewaretoken=QD3t15dOtiDPDfhIScjFJWOD3biW1mjIiBjwBRH89kvKYuAaOYAnxFSeripoDSB6&fullysampled=unknown&uploader=&project=Stanford+2D+FSE&anatomy=&references=&comments=&tags=&funding_support=&system_vendor=&system_model=&protocol_name=&coil_name=&sequence_type=&uuid=efac30a2-8b30-48e8-9a7c-aad2a05a46df


Alle filer kan køres separat.
Uddybende forklaringer og instrukser er angivet i filerne.

"Rekonstruktioner.py" laver rekonstruktioner for både zerofill og least squares med regulariseringsterm.

"Rekonstruktioner_noise.py" laver rekonstruktioner, med tilføjet støj, for både zerofill og least squares with penalty term.

"RMSE_vs_sampling.py" laver en graf over RMSE for mange forskellige samplingsprocenter. Grafen kan laves, hvor der både samples inde fra og ude fra.

"RMSE_vs_sampling_leastsquares.py" laver et plot med grafer over RMSE mod samplingsprocent for 3 forskellige delta-værdier.

"Delta_bestemmelse.py" laver en graf over RMSE for forskellige delta-værdi til en least squares rekonstruktion uden ekstra støj.

"Delta_bestemmelse_noise.py" laver en graf over RMSE mod delta-værdier for en least squares rekonstruktion med ekstra støj.

"RMSE_vs_noise.py" laver et plot med grafer over RMSE for forskellige støjniveauer for både zerofill og least squares, hvor den løbende bedste delta-værdi er brugt til least squares.
