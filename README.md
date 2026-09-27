# RadIA — Sistema de detección y reporte de hallazgos radiológicos con interoperabilidad clínica

Proyecto de portfolio orientado a roles de IA en imagen médica (ej. Floy, y healthtech/medtech
en general). Construido en fases, cada una es un hito demostrable en entrevistas técnicas.

## Fases

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1. Modelo base | CNN (transfer learning) para clasificar neumonía en radiografías de tórax | 🚧 en progreso |
| 2. Pipeline DICOM | Ingesta → anonimización → preprocesado → inferencia → informe estructurado | ⏳ pendiente |
| 3. Explicabilidad | Grad-CAM sobre las predicciones del modelo | ⏳ pendiente |
| 4. Interoperabilidad FHIR | Mapeo del informe estructurado a `DiagnosticReport`/`Observation` FHIR | ⏳ pendiente |
| 5. Cumplimiento normativo | Model card + análisis de riesgo EU AI Act / MDR | ⏳ pendiente |

## Por qué este proyecto

Empresas como Floy construyen pipelines que reciben un estudio DICOM, lo preprocesan,
corren varios modelos de IA, y devuelven un informe estructurado integrado en el flujo
del radiólogo — todo bajo normativa de dispositivo médico europea. Este proyecto replica
esa arquitectura a pequeña escala, y añade una capa de interoperabilidad clínica (FHIR)
que es el diferencial real frente a otros candidatos de ML puro.

## Stack

Python, PyTorch, DICOM (pydicom), FHIR (fhir.resources), Grad-CAM, scikit-learn.

## Cómo ejecutar la Fase 1

Ver `phase1_model/README.md`.
