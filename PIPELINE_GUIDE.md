# 🚀 Guía Completa del Pipeline Genérico de Ingesta de Datos

## 📖 Resumen

Has transformado tu pipeline específico de exchange rates en un **pipeline genérico y reutilizable** que puede:

- ✅ Ingestar datos desde **cualquier fuente** (API, SQL, GCS, archivos locales)
- ✅ Subir automáticamente a **BigQuery**
- ✅ Ejecutarse automáticamente con **GitHub Actions**
- ✅ Adaptarse a **cualquier estructura de datos**
- ✅ Ser **reutilizado** por otros proyectos

---

## 🎯 Lo que Cambiamos

### Antes (Específico)
```
exchange_rate_pipeline.py  → Solo exchange rates de CMF Chile
exchange_rate_fetcher.py   → Solo API CMF Chile
```

### Ahora (Genérico)
```
data_ingestion_pipeline.py → Funciona con cualquier fuente
api_data_fetcher.py        → Framework para cualquier API
data_loader.py             → Soporta SQL, GCS, archivos locales
```

---

## 📂 Archivos Principales

### 1. Pipeline Genérico
**[pipelines/data_ingestion_pipeline.py](pipelines/data_ingestion_pipeline.py)**
- Pipeline principal que orquesta toda la ingesta
- Soporta múltiples fuentes de datos
- Modos: incremental, full, backfill

### 2. Fetchers de Datos

**[src/ingestion/api_data_fetcher.py](src/ingestion/api_data_fetcher.py)**
- Clase base `BaseAPIFetcher` para cualquier API
- Implementación `CMFChileAPIFetcher` como ejemplo
- Fácil de extender para nuevas APIs

**[src/ingestion/data_loader.py](src/ingestion/data_loader.py)**
- Carga datos desde SQL, GCS, archivos locales
- Maneja diferentes formatos (CSV, Parquet, JSON, Excel)

**[src/ingestion/bigquery_loader.py](src/ingestion/bigquery_loader.py)**
- Carga datos a BigQuery
- Schema dinámico (se adapta a cualquier estructura)
- Soporta upsert, particionamiento, clustering

### 3. Configuración

**[config/config.yaml](config/config.yaml)**
- Configuración centralizada
- Secciones para API, SQL, Pipeline, GCP

**[config/examples/](config/examples/)**
- Ejemplos pre-configurados para diferentes casos de uso

### 4. Automatización

**[.github/workflows/data_ingestion.yml](.github/workflows/data_ingestion.yml)**
- Workflow genérico de GitHub Actions
- Ejecuta diariamente o manualmente
- Soporta todos los tipos de fuentes

---

## 🚀 Cómo Usar

### Opción 1: Script Helper (Más Fácil)

```bash
# Ver opciones
./run_pipeline.sh

# Exchange rates (ejemplo incluido)
./run_pipeline.sh exchange-rates

# Importar desde SQL
./run_pipeline.sh sql-import

# Importar desde archivo local
./run_pipeline.sh local-import

# Modo interactivo
./run_pipeline.sh custom
```

### Opción 2: Línea de Comandos

```bash
# Desde API (exchange rates)
python pipelines/data_ingestion_pipeline.py \
  --source api \
  --mode incremental

# Desde SQL
python pipelines/data_ingestion_pipeline.py \
  --source sql \
  --query "SELECT * FROM mi_tabla" \
  --environment prod \
  --mode full

# Desde archivo local
python pipelines/data_ingestion_pipeline.py \
  --source local \
  --file-path data/raw/datos.csv \
  --file-format csv \
  --mode full

# Desde GCS
python pipelines/data_ingestion_pipeline.py \
  --source gcs \
  --blob-path mi-bucket/datos/archivo.parquet \
  --file-format parquet
```

### Opción 3: GitHub Actions (Automático)

1. Sube el código a GitHub
2. Configura los secretos
3. Ve a **Actions** → **Data Ingestion Pipeline**
4. Click **Run workflow**
5. Selecciona:
   - Source type (api, sql, gcs, local)
   - Mode (incremental, full, backfill)

---

## 🔧 Agregar Tu Propia Fuente de Datos

### Ejemplo: API de Clima

**1. Crea tu fetcher**

`src/ingestion/weather_api_fetcher.py`:
```python
from src.ingestion.api_data_fetcher import BaseAPIFetcher
import pandas as pd
import json

class WeatherAPIFetcher(BaseAPIFetcher):
    BASE_URL = "https://api.weather.com/v1"

    def build_url(self, city, **kwargs):
        return f"{self.BASE_URL}/weather/{city}?key={self.api_key}"

    def parse_response(self, response_data, **kwargs):
        data = json.loads(response_data)
        return pd.DataFrame([data])

    def fetch(self, city):
        url = self.build_url(city)
        response = self.fetch_data(url)
        df = self.parse_response(response)
        return self.prepare_for_bigquery(df)
```

**2. Registra en el pipeline**

En `pipelines/data_ingestion_pipeline.py`, función `fetch_from_api()`:
```python
if api_type == 'weather':
    from src.ingestion.weather_api_fetcher import WeatherAPIFetcher
    fetcher = WeatherAPIFetcher(api_key=api_key)
    city = api_config.get('city', 'Santiago')
    df = fetcher.fetch(city=city)
```

**3. Configura**

`config/config.yaml`:
```yaml
api:
  type: "weather"
  city: "Santiago"

pipeline:
  dataset:
    dataset_id: "weather_data"
    table_id: "daily_weather"
    merge_key: "date"
```

**4. Ejecuta**

```bash
python pipelines/data_ingestion_pipeline.py --source api --mode incremental
```

---

## 📊 Configuración por Caso de Uso

### Exchange Rates (Incluido)

```yaml
api:
  type: "cmf_chile"
  currencies: ["usd", "eur", "uf"]

pipeline:
  dataset:
    dataset_id: "exchange_rates"
    table_id: "exchange_rates"
    partition_field: "Fecha"
    merge_key: "Fecha"
```

### Datos de Ventas desde SQL

```yaml
sql:
  driver: "{ODBC Driver 17 for SQL Server}"
  server_prod: "tu-servidor.database.windows.net"
  database: "ventas_db"
  username: "usuario"

pipeline:
  dataset:
    dataset_id: "sales_data"
    table_id: "daily_sales"
    partition_field: "fecha_venta"
    merge_key: "venta_id"
```

### Logs desde GCS

```yaml
gcp:
  project_id: "mi-proyecto"
  region: "us-central1"

pipeline:
  dataset:
    dataset_id: "logs"
    table_id: "application_logs"
    partition_field: "timestamp"
    cluster_fields: ["severity", "service"]
```

---

## 🎮 Modos de Ejecución

### Incremental
- **Qué hace**: Obtiene datos recientes (últimos 2 días)
- **Cuándo usar**: Ejecución diaria automática
- **Escritura**: Upsert (actualiza si existe, inserta si no)

```bash
python pipelines/data_ingestion_pipeline.py --source api --mode incremental
```

### Full
- **Qué hace**: Obtiene todos los datos históricos
- **Cuándo usar**: Carga inicial o resetear tabla
- **Escritura**: Truncate + Insert (borra todo y recarga)

```bash
python pipelines/data_ingestion_pipeline.py --source api --mode full
```

### Backfill
- **Qué hace**: Obtiene N días específicos
- **Cuándo usar**: Llenar huecos en los datos
- **Escritura**: Upsert

```bash
python pipelines/data_ingestion_pipeline.py \
  --source api \
  --mode backfill \
  --backfill-days 30
```

---

## 🔐 Configuración de Secretos

### Local (testing)

`config/secrets.yaml`:
```yaml
sql:
  password_prod: "tu-password-prod"
  password_dev: "tu-password-dev"

api:
  api_key: "tu-api-key"

gcp:
  credentials_path: "config/gcp-key.json"
```

### GitHub Actions (producción)

En GitHub: `Settings → Secrets and variables → Actions`

Agregar:
- `GCP_PROJECT_ID`: Tu proyecto de GCP
- `GCP_CREDENTIALS`: JSON completo de service account
- `API_KEY`: Tu API key
- `SQL_PASSWORD_PROD`: Password SQL producción
- `SQL_PASSWORD_DEV`: Password SQL desarrollo

---

## 📈 Schema Dinámico

El pipeline crea automáticamente tablas basándose en tus datos:

```python
# Tus datos
df = pd.DataFrame({
    'id': [1, 2, 3],
    'nombre': ['Ana', 'Juan', 'Pedro'],
    'monto': [100.5, 200.3, 150.8],
    'fecha': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
})

# BigQuery crea automáticamente:
# CREATE TABLE mi_tabla (
#   id INTEGER,
#   nombre STRING,
#   monto FLOAT64,
#   fecha DATE,
#   ingestion_timestamp TIMESTAMP,
#   data_source STRING
# )
```

---

## 🛠️ Debugging

### Ver logs

```bash
# Logs del último run
cat logs/data_ingestion_pipeline.log

# Logs en tiempo real
tail -f logs/data_ingestion_pipeline.log
```

### Test local antes de GitHub Actions

```bash
# 1. Configura credenciales
export GOOGLE_APPLICATION_CREDENTIALS="config/gcp-key.json"

# 2. Ejecuta
python pipelines/data_ingestion_pipeline.py --source api --mode incremental

# 3. Verifica en BigQuery
bq query --use_legacy_sql=false \
  'SELECT * FROM `proyecto.dataset.tabla` LIMIT 10'
```

### Errores comunes

**"API key not found"**
```bash
# Verifica secrets.yaml
cat config/secrets.yaml | grep api_key
```

**"Module not found"**
```bash
# Instala dependencias
pip install -r requirements.txt
```

**"Permission denied" en BigQuery**
```bash
# Verifica permisos del service account
gcloud projects get-iam-policy tu-proyecto
```

---

## 💡 Tips y Mejores Prácticas

### 1. Testing incremental primero
Siempre prueba en modo incremental antes de full:
```bash
python pipelines/data_ingestion_pipeline.py --source api --mode incremental
```

### 2. Usa particionamiento
Para tablas grandes, siempre particiona:
```yaml
pipeline:
  dataset:
    partition_field: "fecha"  # Campo DATE o TIMESTAMP
```

### 3. Define merge_key
Para evitar duplicados:
```yaml
pipeline:
  dataset:
    merge_key: "id"  # Tu campo único/primary key
```

### 4. Backfill conservador
No hagas backfills muy grandes de una vez:
```bash
# Mejor: 30 días a la vez
--backfill-days 30

# En lugar de: años completos
--mode full
```

### 5. Monitorea costos
```bash
# Ver uso de BigQuery
bq show --format=prettyjson proyecto:dataset.tabla | grep numBytes
```

---

## 📚 Recursos

- **[README.md](README.md)**: Documentación completa
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Guía de despliegue detallada
- **[config/examples/](config/examples/)**: Ejemplos de configuración
- **GitHub Actions**: Revisa los logs en la pestaña Actions

---

## 🎓 Casos de Uso Ejemplo

### 1. Monitoreo de Precios (API)
```bash
# Configurar API de precios
# Ejecutar diariamente
# Almacenar histórico en BigQuery
```

### 2. ETL desde Data Warehouse (SQL)
```bash
# Extraer de SQL Server/PostgreSQL
# Transformar en Python
# Cargar a BigQuery
```

### 3. Procesar Archivos Subidos (GCS)
```bash
# Usuarios suben CSV a GCS
# Pipeline detecta nuevos archivos
# Ingesta automática a BigQuery
```

### 4. Migración de Datos (Local)
```bash
# Archivos legacy en servidor
# Pipeline lee y transforma
# Migra a BigQuery
```

---

## ✅ Checklist de Implementación

- [ ] Configurar GCP project
- [ ] Crear service account con permisos BigQuery
- [ ] Editar `config/config.yaml` con tu configuración
- [ ] Crear `config/secrets.yaml` con tus credenciales
- [ ] Probar localmente: `./run_pipeline.sh exchange-rates`
- [ ] Subir a GitHub
- [ ] Configurar GitHub Secrets
- [ ] Probar GitHub Actions manualmente
- [ ] Verificar datos en BigQuery
- [ ] Configurar schedule (si necesario)
- [ ] Documentar tu caso de uso específico

---

## 🚀 ¡Todo Listo!

Tu pipeline ahora es:
- ✅ **Genérico**: Funciona con cualquier dato
- ✅ **Escalable**: Schema dinámico
- ✅ **Automatizado**: GitHub Actions
- ✅ **Reutilizable**: Otros pueden usarlo
- ✅ **Documentado**: Fácil de mantener

**Próximos pasos sugeridos:**
1. Agrega tu primera fuente de datos personalizada
2. Configura notificaciones (Slack/Email) en GitHub Actions
3. Agrega validaciones de calidad de datos
4. Crea dashboards en Looker Studio con tus datos

¡Éxito con tu pipeline! 🎉
