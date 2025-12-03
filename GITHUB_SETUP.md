# 🚀 GitHub Actions Setup - Exchange Rate Pipeline

## Secretos Necesarios en GitHub

Para que el pipeline funcione en GitHub Actions, necesitas configurar **solo 3 secretos**:

### 1. Ve a GitHub Secrets

```
Tu Repositorio → Settings → Secrets and variables → Actions → New repository secret
```

### 2. Agrega estos 3 secretos:

---

#### Secret 1: `GCP_PROJECT_ID`

**Valor:** Tu ID de proyecto en Google Cloud

```
Ejemplo: my-exchange-rate-project
```

**Cómo obtenerlo:**
```bash
# Si ya tienes gcloud configurado
gcloud config get-value project

# O ve a Google Cloud Console
# https://console.cloud.google.com/
# El ID está en la parte superior
```

---

#### Secret 2: `GCP_CREDENTIALS`

**Valor:** El contenido **completo** del archivo JSON de service account

**Cómo obtenerlo:**

```bash
# 1. Ejecuta el script de setup
./setup_github_actions.sh

# 2. Esto creará el archivo: gcp-key.json

# 3. Copia TODO el contenido:
cat gcp-key.json

# 4. Pega TODO en el secret GCP_CREDENTIALS
```

**El JSON se ve así:**
```json
{
  "type": "service_account",
  "project_id": "tu-proyecto-123",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "exchange-rate-pipeline@tu-proyecto-123.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  ...
}
```

⚠️ **IMPORTANTE:** Copia el JSON completo, incluyendo las llaves `{ }` y todos los saltos de línea.

---

#### Secret 3: `CMF_API_KEY`

**Valor:** Tu API key de CMF Chile

```
Ejemplo: f0b4714b4b79303883d4360a8193f699f8bb96b0
```

**Cómo obtenerlo:**

1. Ve a: https://api.cmfchile.cl/
2. Regístrate (es gratis)
3. Obtén tu API key
4. Copia y pega en el secret

**Nota:** Ya tienes una en `config/secrets.yaml`, puedes usar esa misma:
```bash
cat config/secrets.yaml | grep api_key
```

---

## ✅ Resumen de Secretos

| Secret Name | Descripción | Ejemplo |
|-------------|-------------|---------|
| `GCP_PROJECT_ID` | ID de tu proyecto GCP | `my-project-123` |
| `GCP_CREDENTIALS` | JSON completo del service account | `{"type":"service_account",...}` |
| `CMF_API_KEY` | API key de CMF Chile | `f0b4714b...` |

---

## 📋 Checklist

- [ ] Secret `GCP_PROJECT_ID` agregado
- [ ] Secret `GCP_CREDENTIALS` agregado (JSON completo)
- [ ] Secret `CMF_API_KEY` agregado
- [ ] Código subido a GitHub (`git push`)
- [ ] Workflow aparece en pestaña Actions

---

## 🎮 Cómo Usar el Workflow

### Ejecución Automática
- Se ejecuta **diariamente a las 9:00 AM UTC**
- Modo: incremental (últimos 2 días)
- No necesitas hacer nada

### Ejecución Manual

1. Ve a tu repositorio en GitHub
2. Click en pestaña **Actions**
3. Click en **Exchange Rate Data Ingestion** (izquierda)
4. Click botón **Run workflow** (derecha)
5. Selecciona:
   - **Mode**:
     - `incremental` - Últimos 2 días (recomendado)
     - `full` - Todos los datos desde 1990
     - `backfill` - N días específicos
   - **Backfill days**: Solo si elegiste `backfill` (ej: 30)
6. Click **Run workflow**

---

## 🔍 Ver Resultados

### En GitHub Actions

1. Ve a Actions
2. Click en el run que se ejecutó
3. Click en el job `ingest_data`
4. Verás todos los logs en tiempo real

### Descargar Logs

Si el workflow falla o quieres ver logs detallados:
1. Ve al run en Actions
2. Scroll hasta abajo
3. En "Artifacts" → download `pipeline-logs`

### En BigQuery

```bash
# Desde terminal
bq query --use_legacy_sql=false \
'SELECT * FROM `tu-proyecto.data_ingestion.raw_data`
 ORDER BY Fecha DESC LIMIT 10'
```

O ve a: https://console.cloud.google.com/bigquery

Navega a: `tu-proyecto` → `data_ingestion` → `raw_data`

---

## 🐛 Troubleshooting

### Error: "Bad credentials"
→ Verifica que `GCP_CREDENTIALS` tenga el JSON completo

```bash
# El JSON debe empezar con {
# y terminar con }
# Y tener todas las comillas y comas correctas
```

### Error: "Project not found"
→ Verifica `GCP_PROJECT_ID`

```bash
# Debe ser el ID exacto del proyecto, no el nombre
# Ejemplo correcto: my-project-123
# Ejemplo incorrecto: My Project
```

### Error: "API key not found" o "Invalid API key"
→ Verifica `CMF_API_KEY`

```bash
# Debe ser solo el key, sin espacios
# Ejemplo: f0b4714b4b79303883d4360a8193f699f8bb96b0
```

### Error: "Permission denied" en BigQuery
→ El service account necesita permisos

```bash
# Ejecuta nuevamente
./setup_github_actions.sh

# O manualmente:
gcloud projects add-iam-policy-binding tu-proyecto \
  --member="serviceAccount:exchange-rate-pipeline@tu-proyecto.iam.gserviceaccount.com" \
  --role="roles/bigquery.admin"
```

---

## 🔐 Seguridad

### ✅ Buenas Prácticas

1. **NUNCA** hagas commit de `config/secrets.yaml`
2. **NUNCA** hagas commit de `gcp-key.json`
3. Los secretos en GitHub están encriptados
4. Solo tú y los admins del repo pueden verlos
5. En los logs, los secretos aparecen como `***`

### ✅ Ya Configurado en .gitignore

```bash
# Estos archivos NO se suben a GitHub
config/secrets.yaml
config/gcp-key.json
gcp-key.json
*.json  # (en config/)
```

---

## 📝 Ejemplo Completo

```bash
# 1. Configurar GCP
./setup_github_actions.sh
# → Genera gcp-key.json
# → Te muestra el GCP_PROJECT_ID

# 2. Copiar valores
cat gcp-key.json  # → Copiar para GCP_CREDENTIALS
cat config/secrets.yaml | grep api_key  # → Copiar para CMF_API_KEY

# 3. Ir a GitHub
# Settings → Secrets and variables → Actions

# 4. Agregar 3 secretos:
# - GCP_PROJECT_ID: my-project-123
# - GCP_CREDENTIALS: {JSON completo}
# - CMF_API_KEY: f0b4714b...

# 5. Push a GitHub
git add .
git commit -m "Configure exchange rate pipeline"
git push origin main

# 6. Probar en GitHub Actions
# Actions → Exchange Rate Data Ingestion → Run workflow
```

---

## 🎯 Workflow Simplificado

El workflow **siempre** usa:
- ✅ Source: API (CMF Chile)
- ✅ Data: Exchange rates (USD, EUR, UF)
- ✅ Destino: BigQuery

**No necesitas especificar:**
- ❌ Source type (siempre es API)
- ❌ SQL credentials (no se usa SQL)
- ❌ GCS paths (no se usa GCS)
- ❌ Local files (no se usa local)

**Solo configuras:**
- ✅ Mode (incremental/full/backfill)
- ✅ Backfill days (si usas backfill)

---

## 📊 Datos Generados

El workflow crea en BigQuery:

```
Dataset: data_ingestion
Table: raw_data

Columns:
- Fecha (DATE)          - Fecha del exchange rate
- usdclp_obs (FLOAT64)  - USD a CLP
- eurclp_obs (FLOAT64)  - EUR a CLP
- ufclp (FLOAT64)       - UF en CLP
- ingestion_timestamp   - Cuándo se ingirió
- data_source           - "CMF_Chile_API"
```

---

## ✅ Todo Listo

Después de configurar los 3 secretos:

1. ✅ Workflow se ejecuta diariamente automáticamente
2. ✅ Puedes ejecutarlo manualmente cuando quieras
3. ✅ Logs disponibles en Actions
4. ✅ Datos en BigQuery actualizados
5. ✅ Sin duplicados (upsert automático)

**Siguiente paso:** Probar manualmente en GitHub Actions

```
Actions → Exchange Rate Data Ingestion → Run workflow → Mode: incremental
```

¡Listo! 🚀
