#!/usr/bin/env bash

# Paperless AI Classifier - Proxmox VE Installer
# https://github.com/mwckr/paperless-ai-classifier
#
# SAFETY FIRST: This script is designed to never affect existing resources.
# - All operations are validated before execution
# - Cleanup handlers ensure no orphaned resources on failure
# - Existing VMs/containers/storage are never modified
#
# Run: bash -c "$(wget -qLO - https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main/pve-install.sh)"

# =============================================================================
# STRICT MODE - but we handle errors ourselves
# =============================================================================

set -o pipefail  # Catch pipe failures
# NOT using set -e because we need granular error handling

# =============================================================================
# GLOBAL STATE - for cleanup tracking
# =============================================================================

OLLAMA_CTID=""
API_CTID=""
OLLAMA_CREATED=false
API_CREATED=false
SCRIPT_SUCCESS=false
LOG_FILE="/tmp/paperless-ai-install-$$.log"

# =============================================================================
# COLORS AND FORMATTING
# =============================================================================

RD=$'\033[01;31m'
YW=$'\033[33m'
GN=$'\033[1;92m'
CL=$'\033[m'
BL=$'\033[36m'
OR=$'\033[38;5;208m'
DGN=$'\033[32m'
BGN=$'\033[4;92m'
BOLD=$'\033[1m'

CM="${GN}✓${CL}"
CROSS="${RD}✗${CL}"
INFO="${BL}ℹ${CL}"
WARN="${OR}⚠${CL}"

TAB="  "

# =============================================================================
# LOGGING
# =============================================================================

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

msg_info() {
  echo -e " ${INFO} ${YW}${1}${CL}"
  log "INFO: $1"
}

msg_ok() {
  echo -e " ${CM} ${GN}${1}${CL}"
  log "OK: $1"
}

msg_warn() {
  echo -e " ${WARN} ${OR}${1}${CL}"
  log "WARN: $1"
}

msg_error() {
  echo -e " ${CROSS} ${RD}${1}${CL}"
  log "ERROR: $1"
}

die() {
  msg_error "$1"
  echo -e "\n${YW}Log file: ${LOG_FILE}${CL}"
  exit 1
}

# =============================================================================
# CLEANUP HANDLER - Critical for safety
# =============================================================================

cleanup() {
  local exit_code=$?
  
  # Don't cleanup if script succeeded
  if [[ "$SCRIPT_SUCCESS" == "true" ]]; then
    log "Script completed successfully, no cleanup needed"
    return 0
  fi
  
  log "Cleanup triggered (exit code: $exit_code)"
  
  echo ""
  echo -e "${OR}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
  echo -e "${OR}              INSTALLATION INTERRUPTED                      ${CL}"
  echo -e "${OR}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
  echo ""
  
  # Check what was created and offer cleanup
  local created_items=()
  
  if [[ "$API_CREATED" == "true" && -n "$API_CTID" ]]; then
    created_items+=("CT $API_CTID (API)")
  fi
  
  if [[ "$OLLAMA_CREATED" == "true" && -n "$OLLAMA_CTID" ]]; then
    created_items+=("CT $OLLAMA_CTID (Ollama)")
  fi
  
  if [[ ${#created_items[@]} -eq 0 ]]; then
    echo -e "${GN}No containers were created. System unchanged.${CL}"
    echo -e "\nLog file: ${LOG_FILE}"
    return 0
  fi
  
  echo -e "${YW}The following containers were created before the interruption:${CL}"
  for item in "${created_items[@]}"; do
    echo -e "${TAB}• $item"
  done
  echo ""
  
  # Interactive cleanup prompt (only if terminal is interactive)
  if [[ -t 0 ]]; then
    echo -e "${YW}Would you like to remove these containers?${CL}"
    echo -e "${TAB}${GN}y${CL} = Remove (clean slate for retry)"
    echo -e "${TAB}${RD}n${CL} = Keep (manual cleanup needed)"
    echo ""
    read -rp "Remove created containers? [y/N]: " cleanup_choice
    
    if [[ "$cleanup_choice" =~ ^[Yy]$ ]]; then
      echo ""
      
      if [[ "$API_CREATED" == "true" && -n "$API_CTID" ]]; then
        msg_info "Stopping CT $API_CTID..."
        pct stop "$API_CTID" 2>/dev/null || true
        sleep 2
        msg_info "Destroying CT $API_CTID..."
        if pct destroy "$API_CTID" --purge 2>/dev/null; then
          msg_ok "Removed CT $API_CTID"
        else
          msg_warn "Could not remove CT $API_CTID - manual cleanup may be needed"
        fi
      fi
      
      if [[ "$OLLAMA_CREATED" == "true" && -n "$OLLAMA_CTID" ]]; then
        msg_info "Stopping CT $OLLAMA_CTID..."
        pct stop "$OLLAMA_CTID" 2>/dev/null || true
        sleep 2
        msg_info "Destroying CT $OLLAMA_CTID..."
        if pct destroy "$OLLAMA_CTID" --purge 2>/dev/null; then
          msg_ok "Removed CT $OLLAMA_CTID"
        else
          msg_warn "Could not remove CT $OLLAMA_CTID - manual cleanup may be needed"
        fi
      fi
      
      echo ""
      msg_ok "Cleanup complete. You can safely re-run the installer."
    else
      echo ""
      echo -e "${YW}Containers kept. To manually remove:${CL}"
      [[ "$API_CREATED" == "true" ]] && echo -e "${TAB}pct stop $API_CTID && pct destroy $API_CTID --purge"
      [[ "$OLLAMA_CREATED" == "true" ]] && echo -e "${TAB}pct stop $OLLAMA_CTID && pct destroy $OLLAMA_CTID --purge"
    fi
  else
    echo -e "${YW}Non-interactive mode - containers NOT removed.${CL}"
    echo -e "To manually remove:"
    [[ "$API_CREATED" == "true" ]] && echo -e "${TAB}pct stop $API_CTID && pct destroy $API_CTID --purge"
    [[ "$OLLAMA_CREATED" == "true" ]] && echo -e "${TAB}pct stop $OLLAMA_CTID && pct destroy $OLLAMA_CTID --purge"
  fi
  
  echo ""
  echo -e "Log file: ${LOG_FILE}"
}

# Set trap for cleanup - catches Ctrl+C, errors, and normal exit
trap cleanup EXIT

# =============================================================================
# APP DEFAULTS
# =============================================================================

APP="Paperless-AI-Classifier"

# Ollama LXC defaults
var_ollama_cpu="10"
var_ollama_ram="16384"
var_ollama_disk="50"
var_ollama_hostname="ollama"

# API LXC defaults
var_api_cpu="2"
var_api_ram="2048"
var_api_disk="8"
var_api_hostname="paperless-ai"

# Model defaults
var_model="ministral-3:14b"
var_model_ram="16384"

# Network defaults
var_bridge="vmbr0"
var_vlan=""

# Common
var_os="debian"
var_version="12"
var_unprivileged="1"
var_timezone="host"

# Thresholds
RAM_WARN_PERCENT=80
RAM_CRIT_PERCENT=95
STORAGE_MIN_GB=5

# Paperless (set during configuration)
PAPERLESS_URL=""
PAPERLESS_TOKEN=""

# Storage (set during selection)
TEMPLATE_STORAGE=""
CONTAINER_STORAGE=""
TEMPLATE=""

# IPs (set during setup)
OLLAMA_IP=""
API_IP=""

# =============================================================================
# HEADER
# =============================================================================

header_info() {
  clear
  cat <<"EOF"
    ____                        __                   ___    ____
   / __ \____ _____  ___  _____/ /__  __________    /   |  /  _/
  / /_/ / __ `/ __ \/ _ \/ ___/ / _ \/ ___/ ___/   / /| |  / /  
 / ____/ /_/ / /_/ /  __/ /  / /  __(__  |__  )   / ___ |_/ /   
/_/    \__,_/ .___/\___/_/  /_/\___/____/____/   /_/  |_/___/   
           /_/                                                   
         ________                _ _____          
        / ____/ /___ ___________(_) __(_)__  _____
       / /   / / __ `/ ___/ ___/ / /_/ / _ \/ ___/
      / /___/ / /_/ (__  |__  ) / __/ /  __/ /    
      \____/_/\__,_/____/____/_/_/ /_/\___/_/     

EOF
  echo -e "${BL}Proxmox VE Installer${CL}"
  echo -e "${DGN}https://github.com/mwckr/paperless-ai-classifier${CL}"
  echo ""
}

# =============================================================================
# SAFE EXIT
# =============================================================================

exit_script() {
  SCRIPT_SUCCESS=true  # Prevent cleanup from running
  clear
  echo -e "${YW}Installation cancelled by user. No changes were made.${CL}"
  exit 0
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

# Validate container ID is available (not in use anywhere)
validate_container_id() {
  local ctid="$1"
  
  # Must be numeric
  if ! [[ "$ctid" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  
  # Must be in valid range (100-999999999)
  if [[ "$ctid" -lt 100 ]] || [[ "$ctid" -gt 999999999 ]]; then
    return 1
  fi
  
  # Check Proxmox config files (most reliable check)
  if [[ -f "/etc/pve/qemu-server/${ctid}.conf" ]]; then
    log "CTID $ctid: VM config exists"
    return 1
  fi
  
  if [[ -f "/etc/pve/lxc/${ctid}.conf" ]]; then
    log "CTID $ctid: LXC config exists"
    return 1
  fi
  
  # Check LVM (some storage backends use CTID in LV names)
  if command -v lvs &>/dev/null; then
    if lvs --noheadings -o lv_name 2>/dev/null | grep -qE "(^|[-_])${ctid}($|[-_])"; then
      log "CTID $ctid: Found in LVM"
      return 1
    fi
  fi
  
  return 0
}

# Find next available container ID with safety limit
get_valid_container_id() {
  local start_id="${1:-100}"
  local max_iterations=1000
  local iterations=0
  
  # Get cluster's suggested next ID as starting point
  local suggested
  suggested=$(pvesh get /cluster/nextid 2>/dev/null) || suggested="$start_id"
  
  # Ensure it's numeric
  if ! [[ "$suggested" =~ ^[0-9]+$ ]]; then
    suggested="$start_id"
  fi
  
  # Use the higher of suggested or start_id (important for second container)
  local ctid="$suggested"
  if [[ "$start_id" -gt "$ctid" ]]; then
    ctid="$start_id"
  fi
  
  while ! validate_container_id "$ctid"; do
    ctid=$((ctid + 1))
    iterations=$((iterations + 1))
    
    if [[ $iterations -ge $max_iterations ]]; then
      log "ERROR: Could not find available CTID after $max_iterations attempts"
      echo ""
      return 1
    fi
  done
  
  echo "$ctid"
  return 0
}

# Validate hostname format
validate_hostname() {
  local hostname="$1"
  
  [[ -z "$hostname" ]] && return 1
  [[ ${#hostname} -gt 63 ]] && return 1
  
  # RFC 1123: alphanumeric and hyphens, can't start/end with hyphen
  if [[ ! "$hostname" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] && [[ ! "$hostname" =~ ^[a-z0-9]$ ]]; then
    return 1
  fi
  
  return 0
}

# Validate network bridge exists and is up
validate_bridge() {
  local bridge="$1"
  
  [[ -z "$bridge" ]] && return 1
  
  # Check interface exists
  if ! ip link show "$bridge" &>/dev/null; then
    log "Bridge $bridge: interface not found"
    return 1
  fi
  
  return 0
}

# Validate VLAN tag
validate_vlan_tag() {
  local vlan="$1"
  
  # Empty is OK (no VLAN)
  [[ -z "$vlan" ]] && return 0
  
  # Must be numeric 1-4094
  if ! [[ "$vlan" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  
  if [[ "$vlan" -lt 1 ]] || [[ "$vlan" -gt 4094 ]]; then
    return 1
  fi
  
  return 0
}

# Validate numeric input
validate_numeric() {
  local value="$1"
  local min="${2:-1}"
  local max="${3:-999999}"
  
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  
  if [[ "$value" -lt "$min" ]] || [[ "$value" -gt "$max" ]]; then
    return 1
  fi
  
  return 0
}

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

get_total_ram_mb() {
  local ram
  ram=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null) || ram=0
  echo "${ram:-0}"
}

get_allocated_ram_mb() {
  local total=0
  local vmid mem
  
  # LXC containers
  while IFS= read -r vmid; do
    [[ -z "$vmid" ]] && continue
    mem=$(pct config "$vmid" 2>/dev/null | awk '/^memory:/ {print $2}')
    total=$((total + ${mem:-0}))
  done < <(pct list 2>/dev/null | awk 'NR>1 {print $1}')
  
  # VMs
  while IFS= read -r vmid; do
    [[ -z "$vmid" ]] && continue
    mem=$(qm config "$vmid" 2>/dev/null | awk '/^memory:/ {print $2}')
    total=$((total + ${mem:-0}))
  done < <(qm list 2>/dev/null | awk 'NR>1 {print $1}')
  
  echo "$total"
}

get_total_cores() {
  nproc 2>/dev/null || echo "1"
}

get_storage_info() {
  local storage="$1"
  local info
  
  info=$(pvesm status -storage "$storage" 2>/dev/null | awk 'NR>1 {
    total_gb = int($4/1024/1024)
    avail_gb = int($5/1024/1024)
    printf "%d %d", total_gb, avail_gb
  }')
  
  # Return "0 0" if failed
  echo "${info:-0 0}"
}

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

preflight_checks() {
  header_info
  echo -e "${BL}Running pre-flight checks...${CL}\n"
  log "Starting pre-flight checks"
  
  local errors=0
  local warnings=0
  
  # Root check
  if [[ $EUID -eq 0 ]]; then
    msg_ok "Running as root"
  else
    msg_error "This script must be run as root"
    exit 1
  fi
  
  # Proxmox check
  if command -v pveversion &>/dev/null; then
    local pve_ver
    pve_ver=$(pveversion | cut -d'/' -f2)
    msg_ok "Proxmox VE $pve_ver detected"
    log "PVE version: $pve_ver"
  else
    msg_error "This script must be run on a Proxmox VE host"
    exit 1
  fi
  
  # Required commands
  local cmds=("pct" "pvesm" "curl" "wget" "awk" "grep")
  for cmd in "${cmds[@]}"; do
    if command -v "$cmd" &>/dev/null; then
      msg_ok "Command '$cmd' available"
    else
      msg_error "Required command '$cmd' not found"
      ((errors++))
    fi
  done
  
  # Check for whiptail (used for menus)
  if ! command -v whiptail &>/dev/null; then
    msg_warn "whiptail not found - will use basic prompts"
    ((warnings++))
  else
    msg_ok "Command 'whiptail' available"
  fi
  
  # Network bridge
  if validate_bridge "$var_bridge"; then
    msg_ok "Network bridge '$var_bridge' exists"
  else
    msg_warn "Default bridge '$var_bridge' not found"
    msg_info "Available bridges:"
    ip link show type bridge 2>/dev/null | grep -E "^[0-9]+" | awk -F: '{print "    " $2}' | tr -d ' '
    ((warnings++))
  fi
  
  # Internet connectivity (with timeout)
  if timeout 5 ping -c1 -W3 8.8.8.8 &>/dev/null; then
    msg_ok "Internet connectivity OK"
  else
    msg_warn "No internet connectivity detected"
    ((warnings++))
  fi
  
  # Can we reach ollama.ai?
  if timeout 10 curl -sfL --connect-timeout 5 https://ollama.ai &>/dev/null; then
    msg_ok "Can reach ollama.ai"
  else
    msg_warn "Cannot reach ollama.ai - model download may fail"
    ((warnings++))
  fi
  
  # GitHub reachable?
  if timeout 10 curl -sfL --connect-timeout 5 https://github.com &>/dev/null; then
    msg_ok "Can reach github.com"
  else
    msg_warn "Cannot reach github.com - script download may fail"
    ((warnings++))
  fi
  
  echo ""
  
  if [[ $errors -gt 0 ]]; then
    die "Pre-flight checks failed with $errors error(s). Cannot continue."
  fi
  
  if [[ $warnings -gt 0 ]]; then
    echo -e "${OR}Pre-flight completed with $warnings warning(s)${CL}"
    echo ""
    read -rp "Continue anyway? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
      SCRIPT_SUCCESS=true
      exit 0
    fi
  else
    msg_ok "All pre-flight checks passed"
  fi
  
  log "Pre-flight checks completed"
  sleep 1
}

# =============================================================================
# STORAGE SELECTION
# =============================================================================

select_storage() {
  local content_type="$1"
  local prompt="$2"
  local storage_list
  
  storage_list=$(pvesm status -content "$content_type" 2>/dev/null | awk 'NR>1 {print $1}')
  
  if [[ -z "$storage_list" ]]; then
    die "No storage available for content type: $content_type"
  fi
  
  local count=0
  local -a storages=()
  
  while IFS= read -r storage; do
    [[ -z "$storage" ]] && continue
    storages+=("$storage")
    ((count++))
  done <<< "$storage_list"
  
  # If only one storage, use it automatically
  if [[ $count -eq 1 ]]; then
    STORAGE_RESULT="${storages[0]}"
    log "Auto-selected storage: $STORAGE_RESULT (only option)"
    return 0
  fi
  
  # If whiptail available, use it
  if command -v whiptail &>/dev/null; then
    local -a menu_items=()
    local first=true
    
    for storage in "${storages[@]}"; do
      local info total avail
      info=$(get_storage_info "$storage")
      total=$(echo "$info" | awk '{print $1}')
      avail=$(echo "$info" | awk '{print $2}')
      
      if [[ "$first" == "true" ]]; then
        menu_items+=("$storage" "${avail}GB free of ${total}GB" "ON")
        first=false
      else
        menu_items+=("$storage" "${avail}GB free of ${total}GB" "OFF")
      fi
    done
    
    STORAGE_RESULT=$(whiptail --backtitle "Proxmox VE Helper Scripts" \
      --title "STORAGE SELECTION" \
      --radiolist "$prompt" 16 60 6 \
      "${menu_items[@]}" \
      3>&1 1>&2 2>&3)
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]] || [[ -z "$STORAGE_RESULT" ]]; then
      exit_script
    fi
  else
    # Fallback to basic menu
    echo -e "$prompt"
    local i=1
    for storage in "${storages[@]}"; do
      local info total avail
      info=$(get_storage_info "$storage")
      total=$(echo "$info" | awk '{print $1}')
      avail=$(echo "$info" | awk '{print $2}')
      echo "  $i) $storage (${avail}GB free of ${total}GB)"
      ((i++))
    done
    
    read -rp "Select storage [1-$count]: " choice
    if ! validate_numeric "$choice" 1 "$count"; then
      STORAGE_RESULT="${storages[0]}"
    else
      STORAGE_RESULT="${storages[$((choice-1))]}"
    fi
  fi
  
  log "Selected storage: $STORAGE_RESULT"
  return 0
}

# =============================================================================
# RESOURCE CHECKS
# =============================================================================

check_resources() {
  header_info
  echo -e "${BL}Checking system resources...${CL}\n"
  log "Starting resource checks"
  
  local errors=0
  local warnings=0
  
  # RAM calculations with safety
  local total_ram allocated_ram requested_ram new_allocated ram_percent
  total_ram=$(get_total_ram_mb)
  allocated_ram=$(get_allocated_ram_mb)
  requested_ram=$((var_ollama_ram + var_api_ram))
  new_allocated=$((allocated_ram + requested_ram))
  
  # Prevent division by zero
  if [[ $total_ram -gt 0 ]]; then
    ram_percent=$((new_allocated * 100 / total_ram))
  else
    ram_percent=100
  fi
  
  echo -e "${YW}Memory (RAM):${CL}"
  echo -e "${TAB}Host total:       ${BOLD}${total_ram}MB${CL} ($((total_ram/1024))GB)"
  echo -e "${TAB}Currently used:   ${allocated_ram}MB"
  echo -e "${TAB}Requested:        ${requested_ram}MB (Ollama: ${var_ollama_ram}MB + API: ${var_api_ram}MB)"
  echo -e "${TAB}After install:    ${new_allocated}MB (${ram_percent}% of total)"
  echo ""
  
  if [[ $ram_percent -ge $RAM_CRIT_PERCENT ]]; then
    msg_warn "RAM allocation is ${ram_percent}% of host RAM (overcommit)"
    echo -e "${TAB}${YW}Note: LXC uses memory dynamically - allocated ≠ actually used${CL}"
    ((warnings++))
  elif [[ $ram_percent -ge $RAM_WARN_PERCENT ]]; then
    msg_warn "RAM allocation at ${ram_percent}% - may be tight under heavy load"
    ((warnings++))
  else
    msg_ok "RAM allocation OK (${ram_percent}% of total)"
  fi
  
  # Model RAM check
  if [[ $var_ollama_ram -lt $var_model_ram ]]; then
    msg_warn "Model '$var_model' recommends ${var_model_ram}MB, only ${var_ollama_ram}MB allocated"
    ((warnings++))
  else
    msg_ok "RAM sufficient for model '$var_model'"
  fi
  
  # CPU cores
  local total_cores requested_cores
  total_cores=$(get_total_cores)
  requested_cores=$((var_ollama_cpu + var_api_cpu))
  
  echo ""
  echo -e "${YW}CPU Cores:${CL}"
  echo -e "${TAB}Host total:    ${BOLD}${total_cores}${CL}"
  echo -e "${TAB}Requested:     ${requested_cores} (Ollama: ${var_ollama_cpu} + API: ${var_api_cpu})"
  echo ""
  
  if [[ $requested_cores -gt $total_cores ]]; then
    msg_warn "Overcommitting CPU cores (${requested_cores}/${total_cores}) - this is OK but noted"
    ((warnings++))
  else
    msg_ok "CPU allocation OK (${requested_cores}/${total_cores} cores)"
  fi
  
  # Storage
  local storage_info storage_total storage_avail requested_disk storage_after
  storage_info=$(get_storage_info "$CONTAINER_STORAGE")
  storage_total=$(echo "$storage_info" | awk '{print $1}')
  storage_avail=$(echo "$storage_info" | awk '{print $2}')
  requested_disk=$((var_ollama_disk + var_api_disk))
  storage_after=$((storage_avail - requested_disk))
  
  echo ""
  echo -e "${YW}Storage (${CONTAINER_STORAGE}):${CL}"
  echo -e "${TAB}Total:         ${BOLD}${storage_total}GB${CL}"
  echo -e "${TAB}Available:     ${storage_avail}GB"
  echo -e "${TAB}Requested:     ${requested_disk}GB (Ollama: ${var_ollama_disk}GB + API: ${var_api_disk}GB)"
  echo -e "${TAB}After install: ${storage_after}GB remaining"
  echo ""
  
  if [[ $storage_after -lt 0 ]]; then
    msg_error "Not enough storage space (need ${requested_disk}GB, have ${storage_avail}GB)"
    ((errors++))
  elif [[ $storage_after -lt $STORAGE_MIN_GB ]]; then
    msg_error "Less than ${STORAGE_MIN_GB}GB would remain after install"
    ((errors++))
  else
    msg_ok "Storage allocation OK (${storage_after}GB will remain)"
  fi
  
  # Container IDs - verify they're still available
  echo ""
  echo -e "${YW}Container IDs:${CL}"
  
  if validate_container_id "$OLLAMA_CTID"; then
    msg_ok "CT $OLLAMA_CTID available for Ollama"
  else
    msg_error "CT $OLLAMA_CTID is no longer available"
    ((errors++))
  fi
  
  if validate_container_id "$API_CTID"; then
    msg_ok "CT $API_CTID available for API"
  else
    msg_error "CT $API_CTID is no longer available"
    ((errors++))
  fi
  
  echo ""
  log "Resource check complete: $errors errors, $warnings warnings"
  
  if [[ $errors -gt 0 ]]; then
    echo -e "${RD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
    msg_error "Resource checks failed with $errors error(s)"
    echo ""
    echo -e "${YW}Suggestions:${CL}"
    echo -e "${TAB}• Choose a smaller model (llava:7b needs 8GB RAM)"
    echo -e "${TAB}• Stop other VMs/containers to free resources"
    echo -e "${TAB}• Use Advanced Settings to reduce allocations"
    echo ""
    die "Cannot continue due to resource constraints"
  fi
  
  if [[ $warnings -gt 0 ]]; then
    echo -e "${OR}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
    echo -e "${OR}Resource checks completed with $warnings warning(s)${CL}"
    echo ""
    read -rp "Continue with these warnings? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
      echo -e "\n${YW}Tip: Run again and use Advanced Settings to adjust resources.${CL}"
      SCRIPT_SUCCESS=true
      exit 0
    fi
  else
    echo -e "${GN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
    msg_ok "All resource checks passed"
  fi
  
  sleep 1
}

# =============================================================================
# PAPERLESS CONFIGURATION
# =============================================================================

configure_paperless() {
  header_info
  
  echo -e "${BL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
  echo -e "${BL}              PAPERLESS CONFIGURATION                      ${CL}"
  echo -e "${BL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
  echo ""
  echo -e "${YW}You need a running Paperless-ngx instance and an API token.${CL}"
  echo ""
  echo -e "To get your API token:"
  echo -e "  1. Open Paperless-ngx in your browser"
  echo -e "  2. Click your ${GN}profile icon${CL} (upper right corner)"
  echo -e "  3. Copy your existing token or generate a new one"
  echo ""
  
  # Get URL with validation
  while true; do
    read -rp "Paperless URL (e.g., http://192.168.1.100:8000): " PAPERLESS_URL
    
    # Handle empty/cancel
    if [[ -z "$PAPERLESS_URL" ]]; then
      read -rp "No URL entered. Cancel installation? [y/N]: " cancel
      [[ "$cancel" =~ ^[Yy]$ ]] && exit_script
      continue
    fi
    
    # Remove trailing slash
    PAPERLESS_URL="${PAPERLESS_URL%/}"
    
    # Validate format
    if [[ ! "$PAPERLESS_URL" =~ ^https?://[a-zA-Z0-9][-a-zA-Z0-9.]*[a-zA-Z0-9](:[0-9]+)?$ ]]; then
      msg_warn "Invalid URL format. Expected: http://hostname:port or https://hostname:port"
      continue
    fi
    
    log "Paperless URL set: $PAPERLESS_URL"
    break
  done
  
  # Get Token with validation
  while true; do
    read -rp "Paperless API Token: " PAPERLESS_TOKEN
    
    if [[ -z "$PAPERLESS_TOKEN" ]]; then
      read -rp "No token entered. Cancel installation? [y/N]: " cancel
      [[ "$cancel" =~ ^[Yy]$ ]] && exit_script
      continue
    fi
    
    if [[ ${#PAPERLESS_TOKEN} -lt 10 ]]; then
      msg_warn "Token seems too short (minimum 10 characters)"
      continue
    fi
    
    # Don't log the actual token
    log "Paperless token set (length: ${#PAPERLESS_TOKEN})"
    break
  done
  
  # Test connection - use /api/documents/ which returns 200 directly (no redirects)
  echo ""
  msg_info "Testing connection to Paperless..."
  
  local response
  
  # Test against /api/documents/ - returns 200 on success, 401/403 on auth failure
  response=$(timeout 15 curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Token $PAPERLESS_TOKEN" \
    "$PAPERLESS_URL/api/documents/" \
    --connect-timeout 10 2>/dev/null) || response="000"
  
  log "Paperless test /api/documents/: HTTP $response"
  
  case "$response" in
    200)
      msg_ok "Successfully connected to Paperless"
      ;;
    401|403)
      msg_error "Invalid API token (HTTP $response)"
      echo -e "${YW}Please verify your token and try again.${CL}"
      die "Authentication failed"
      ;;
    302|301)
      # Redirect to login means token not accepted
      msg_warn "Paperless is redirecting to login (HTTP $response)"
      echo -e "${YW}This usually means the token is invalid.${CL}"
      read -rp "Continue anyway? [y/N]: " confirm
      if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        SCRIPT_SUCCESS=true
        exit 0
      fi
      ;;
    000)
      msg_warn "Could not connect to Paperless (timeout or network error)"
      echo -e "${YW}Make sure Paperless is running and accessible from this host.${CL}"
      read -rp "Continue anyway? [y/N]: " confirm
      if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        SCRIPT_SUCCESS=true
        exit 0
      fi
      ;;
    404)
      msg_warn "Paperless responded but API endpoint not found (HTTP 404)"
      echo -e "${YW}The URL might be incorrect or Paperless version incompatible.${CL}"
      read -rp "Continue anyway? [y/N]: " confirm
      if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        SCRIPT_SUCCESS=true
        exit 0
      fi
      ;;
    *)
      msg_warn "Unexpected response from Paperless (HTTP $response)"
      read -rp "Continue anyway? [y/N]: " confirm
      if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        SCRIPT_SUCCESS=true
        exit 0
      fi
      ;;
  esac
  
  sleep 1
}

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

advanced_settings() {
  log "Entering advanced settings"
  
  # Check if whiptail is available
  if ! command -v whiptail &>/dev/null; then
    advanced_settings_basic
    return
  fi
  
  # Model selection
  local model_choice
  model_choice=$(whiptail --backtitle "Proxmox VE Helper Scripts" \
    --title "VISION MODEL" \
    --menu "\nSelect the AI vision model:\n" 16 70 4 \
    "1" "ministral-3:14b (recommended) - Best quality, needs 16GB RAM" \
    "2" "llava:13b - Good quality, needs 12GB RAM" \
    "3" "llava:7b - Faster/lighter, needs 8GB RAM" \
    "4" "Custom model" \
    3>&1 1>&2 2>&3)
  
  # Handle cancel - return to defaults
  if [[ $? -ne 0 ]]; then
    log "Advanced settings cancelled, using defaults"
    return
  fi
  
  case "$model_choice" in
    1)
      var_model="ministral-3:14b"
      var_model_ram="16384"
      var_ollama_ram="16384"
      ;;
    2)
      var_model="llava:13b"
      var_model_ram="12288"
      var_ollama_ram="12288"
      ;;
    3)
      var_model="llava:7b"
      var_model_ram="8192"
      var_ollama_ram="8192"
      ;;
    4)
      var_model=$(whiptail --inputbox "Enter custom model name:" 8 60 "$var_model" 3>&1 1>&2 2>&3) || true
      local custom_ram
      custom_ram=$(whiptail --inputbox "RAM required (MB):" 8 60 "$var_model_ram" 3>&1 1>&2 2>&3) || true
      if validate_numeric "$custom_ram" 1024 131072; then
        var_model_ram="$custom_ram"
        var_ollama_ram="$custom_ram"
      fi
      ;;
  esac
  
  log "Model selected: $var_model (RAM: $var_model_ram)"
  
  # Ollama resources - with validation
  local input
  
  input=$(whiptail --inputbox "Ollama CPU cores:" 8 60 "$var_ollama_cpu" 3>&1 1>&2 2>&3) || true
  validate_numeric "$input" 1 128 && var_ollama_cpu="$input"
  
  input=$(whiptail --inputbox "Ollama RAM (MB):" 8 60 "$var_ollama_ram" 3>&1 1>&2 2>&3) || true
  validate_numeric "$input" 512 262144 && var_ollama_ram="$input"
  
  input=$(whiptail --inputbox "Ollama Disk (GB):" 8 60 "$var_ollama_disk" 3>&1 1>&2 2>&3) || true
  validate_numeric "$input" 10 10000 && var_ollama_disk="$input"
  
  input=$(whiptail --inputbox "Ollama hostname:" 8 60 "$var_ollama_hostname" 3>&1 1>&2 2>&3) || true
  [[ -n "$input" ]] && validate_hostname "${input,,}" && var_ollama_hostname="${input,,}"
  
  # API resources
  input=$(whiptail --inputbox "API CPU cores:" 8 60 "$var_api_cpu" 3>&1 1>&2 2>&3) || true
  validate_numeric "$input" 1 32 && var_api_cpu="$input"
  
  input=$(whiptail --inputbox "API RAM (MB):" 8 60 "$var_api_ram" 3>&1 1>&2 2>&3) || true
  validate_numeric "$input" 256 32768 && var_api_ram="$input"
  
  input=$(whiptail --inputbox "API Disk (GB):" 8 60 "$var_api_disk" 3>&1 1>&2 2>&3) || true
  validate_numeric "$input" 2 1000 && var_api_disk="$input"
  
  input=$(whiptail --inputbox "API hostname:" 8 60 "$var_api_hostname" 3>&1 1>&2 2>&3) || true
  [[ -n "$input" ]] && validate_hostname "${input,,}" && var_api_hostname="${input,,}"
  
  # Network
  input=$(whiptail --inputbox "Network bridge:" 8 60 "$var_bridge" 3>&1 1>&2 2>&3) || true
  [[ -n "$input" ]] && validate_bridge "$input" && var_bridge="$input"
  
  input=$(whiptail --inputbox "VLAN tag (empty for none):" 8 60 "$var_vlan" 3>&1 1>&2 2>&3) || true
  validate_vlan_tag "$input" && var_vlan="$input"
  
  log "Advanced settings completed"
}

advanced_settings_basic() {
  # Fallback for systems without whiptail
  echo ""
  echo -e "${YW}Advanced Settings (press Enter to keep default):${CL}"
  echo ""
  
  local input
  
  echo -e "${BL}Vision Model:${CL}"
  echo "  1) ministral-3:14b (default, 16GB RAM)"
  echo "  2) llava:13b (12GB RAM)"
  echo "  3) llava:7b (8GB RAM)"
  read -rp "Select [1]: " input
  case "$input" in
    2)
      var_model="llava:13b"
      var_model_ram="12288"
      var_ollama_ram="12288"
      ;;
    3)
      var_model="llava:7b"
      var_model_ram="8192"
      var_ollama_ram="8192"
      ;;
  esac
  
  read -rp "Ollama CPU cores [$var_ollama_cpu]: " input
  validate_numeric "$input" 1 128 && var_ollama_cpu="$input"
  
  read -rp "Ollama RAM MB [$var_ollama_ram]: " input
  validate_numeric "$input" 512 262144 && var_ollama_ram="$input"
  
  read -rp "Ollama Disk GB [$var_ollama_disk]: " input
  validate_numeric "$input" 10 10000 && var_ollama_disk="$input"
  
  read -rp "API CPU cores [$var_api_cpu]: " input
  validate_numeric "$input" 1 32 && var_api_cpu="$input"
  
  read -rp "API RAM MB [$var_api_ram]: " input
  validate_numeric "$input" 256 32768 && var_api_ram="$input"
  
  read -rp "Network bridge [$var_bridge]: " input
  [[ -n "$input" ]] && validate_bridge "$input" && var_bridge="$input"
}

# =============================================================================
# TEMPLATE DOWNLOAD
# =============================================================================

download_template() {
  msg_info "Checking for Debian ${var_version} template..."
  log "Looking for Debian $var_version template on $TEMPLATE_STORAGE"
  
  # Check if template already exists
  local existing
  existing=$(pveam list "$TEMPLATE_STORAGE" 2>/dev/null | grep "debian-${var_version}" | head -1 | awk '{print $1}')
  
  if [[ -n "$existing" ]]; then
    TEMPLATE="$existing"
    msg_ok "Template found: $TEMPLATE"
    log "Using existing template: $TEMPLATE"
    return 0
  fi
  
  # Need to download
  msg_info "Downloading Debian ${var_version} template..."
  
  # Update template list
  if ! pveam update &>/dev/null; then
    msg_warn "Could not update template list"
  fi
  
  # Find available template
  local available_template
  available_template=$(pveam available -section system 2>/dev/null | grep "debian-${var_version}" | head -1 | awk '{print $2}')
  
  if [[ -z "$available_template" ]]; then
    die "Could not find Debian ${var_version} template in available list"
  fi
  
  log "Downloading template: $available_template"
  
  # Download with error handling
  if ! pveam download "$TEMPLATE_STORAGE" "$available_template" 2>&1 | tee -a "$LOG_FILE"; then
    die "Failed to download template: $available_template"
  fi
  
  TEMPLATE="${TEMPLATE_STORAGE}:vztmpl/${available_template}"
  
  # Verify download
  if ! pveam list "$TEMPLATE_STORAGE" 2>/dev/null | grep -q "$available_template"; then
    die "Template download verification failed"
  fi
  
  msg_ok "Template downloaded: $TEMPLATE"
  log "Template ready: $TEMPLATE"
}

# =============================================================================
# LXC CREATION - with safety checks
# =============================================================================

create_lxc() {
  local ctid="$1"
  local hostname="$2"
  local cores="$3"
  local ram="$4"
  local disk="$5"
  local desc="$6"
  
  log "Creating $desc LXC: CTID=$ctid hostname=$hostname cores=$cores ram=$ram disk=$disk"
  
  # Final safety check before creation
  if ! validate_container_id "$ctid"; then
    die "Container ID $ctid became unavailable. Another process may have claimed it."
  fi
  
  msg_info "Creating $desc LXC (CT $ctid)..."
  
  local net_config="name=eth0,bridge=${var_bridge},ip=dhcp"
  [[ -n "$var_vlan" ]] && net_config+=",tag=${var_vlan}"
  
  # Create container
  local create_output
  create_output=$(pct create "$ctid" "$TEMPLATE" \
    --hostname "$hostname" \
    --cores "$cores" \
    --memory "$ram" \
    --swap "$((ram / 2))" \
    --rootfs "${CONTAINER_STORAGE}:${disk}" \
    --net0 "$net_config" \
    --unprivileged "$var_unprivileged" \
    --features nesting=1 \
    --onboot 1 \
    --timezone "$var_timezone" \
    --ostype "$var_os" \
    --start 0 2>&1)
  
  local exit_code=$?
  log "pct create output: $create_output"
  
  if [[ $exit_code -ne 0 ]]; then
    msg_error "Failed to create $desc LXC"
    log "pct create failed with exit code $exit_code"
    die "Container creation failed: $create_output"
  fi
  
  # Verify creation
  if [[ ! -f "/etc/pve/lxc/${ctid}.conf" ]]; then
    die "Container $ctid config file not found after creation"
  fi
  
  msg_ok "Created $desc LXC (CT $ctid)"
  
  # Mark as created for cleanup tracking
  if [[ "$desc" == "Ollama" ]]; then
    OLLAMA_CREATED=true
  elif [[ "$desc" == "API" ]]; then
    API_CREATED=true
  fi
}

# =============================================================================
# NETWORK WAIT - with proper timeout handling
# =============================================================================

wait_for_network() {
  local ctid="$1"
  local max_wait="${2:-90}"
  local waited=0
  
  while [[ $waited -lt $max_wait ]]; do
    local ip
    ip=$(pct exec "$ctid" -- hostname -I 2>/dev/null | awk '{print $1}') || true
    
    if [[ -n "$ip" && "$ip" != "127.0.0.1" && "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "$ip"
      return 0
    fi
    
    sleep 2
    waited=$((waited + 2))
  done
  
  log "Network wait timeout for CT $ctid after ${max_wait}s"
  return 1
}

# =============================================================================
# WAIT FOR SERVICE
# =============================================================================

wait_for_service() {
  local url="$1"
  local max_wait="${2:-60}"
  local waited=0
  
  while [[ $waited -lt $max_wait ]]; do
    if timeout 5 curl -sf "$url" &>/dev/null; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  
  return 1
}

# =============================================================================
# OLLAMA SETUP
# =============================================================================

setup_ollama() {
  log "Setting up Ollama LXC"
  
  msg_info "Starting Ollama LXC..."
  if ! pct start "$OLLAMA_CTID" 2>&1 | tee -a "$LOG_FILE"; then
    die "Failed to start Ollama LXC"
  fi
  
  # Wait for boot
  sleep 5
  
  msg_info "Waiting for network..."
  OLLAMA_IP=$(wait_for_network "$OLLAMA_CTID" 90)
  if [[ -z "$OLLAMA_IP" ]]; then
    die "Ollama LXC failed to get IP address. Check DHCP server and network configuration."
  fi
  msg_ok "Ollama IP: $OLLAMA_IP"
  log "Ollama IP: $OLLAMA_IP"
  
  msg_info "Installing Ollama (this may take a few minutes)..."
  
  # Run installation inside container
  local install_output
  install_output=$(pct exec "$OLLAMA_CTID" -- bash -c '
    set -e
    
    # Update and install dependencies
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates >/dev/null 2>&1
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Configure for external access
    mkdir -p /etc/systemd/system/ollama.service.d
    cat > /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
    
    # Enable and start
    systemctl daemon-reload
    systemctl enable ollama >/dev/null 2>&1
    systemctl restart ollama
    
    echo "Ollama installation complete"
  ' 2>&1)
  
  local exit_code=$?
  log "Ollama install output: $install_output"
  
  if [[ $exit_code -ne 0 ]]; then
    die "Ollama installation failed. Check logs: $LOG_FILE"
  fi
  
  msg_ok "Ollama installed"
  
  # Wait for API to be ready
  msg_info "Waiting for Ollama API..."
  if wait_for_service "http://${OLLAMA_IP}:11434/api/tags" 90; then
    msg_ok "Ollama API ready"
  else
    msg_warn "Ollama API not responding yet (may still be starting)"
    log "Ollama API not ready after 90s"
  fi
  
  # Pull model
  msg_info "Pulling model: $var_model"
  echo -e "${TAB}${YW}This may take 10-20 minutes depending on your connection...${CL}"
  
  local pull_output
  pull_output=$(pct exec "$OLLAMA_CTID" -- ollama pull "$var_model" 2>&1)
  exit_code=$?
  log "Model pull output: $pull_output"
  
  if [[ $exit_code -ne 0 ]]; then
    msg_error "Failed to pull model: $var_model"
    echo -e "${YW}You can manually pull the model later:${CL}"
    echo -e "${TAB}pct exec $OLLAMA_CTID -- ollama pull $var_model"
    echo ""
    read -rp "Continue anyway? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
      die "Model pull failed"
    fi
  else
    msg_ok "Model '$var_model' ready"
  fi
  
  log "Ollama setup complete"
}

# =============================================================================
# API SETUP
# =============================================================================

setup_api() {
  log "Setting up API LXC"
  
  msg_info "Starting API LXC..."
  if ! pct start "$API_CTID" 2>&1 | tee -a "$LOG_FILE"; then
    die "Failed to start API LXC"
  fi
  
  sleep 5
  
  msg_info "Waiting for network..."
  API_IP=$(wait_for_network "$API_CTID" 90)
  if [[ -z "$API_IP" ]]; then
    die "API LXC failed to get IP address. Check DHCP server and network configuration."
  fi
  msg_ok "API IP: $API_IP"
  log "API IP: $API_IP"
  
  msg_info "Installing Classifier API..."
  
  local install_output
  install_output=$(pct exec "$API_CTID" -- bash -c '
    set -e
    
    # Update and install dependencies
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip sqlite3 poppler-utils curl ca-certificates >/dev/null 2>&1
    
    # Create directory
    mkdir -p /opt/paperless-classifier
    cd /opt/paperless-classifier
    
    # Download application files
    curl -fsSL https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main/ministral.py -o ministral.py
    curl -fsSL https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main/classifier_api.py -o classifier_api.py
    
    # Verify downloads
    if [[ ! -s ministral.py ]] || [[ ! -s classifier_api.py ]]; then
      echo "ERROR: Failed to download application files"
      exit 1
    fi
    
    # Install Python dependencies
    pip3 install fastapi uvicorn python-dotenv pydantic requests pillow pdf2image --break-system-packages -q
    
    # Create systemd service
    cat > /etc/systemd/system/paperless-classifier.service << EOF
[Unit]
Description=Paperless Document Classifier API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/paperless-classifier
EnvironmentFile=/opt/paperless-classifier/.env
ExecStart=/usr/bin/python3 /opt/paperless-classifier/classifier_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    
    echo "API installation complete"
  ' 2>&1)
  
  local exit_code=$?
  log "API install output: $install_output"
  
  if [[ $exit_code -ne 0 ]]; then
    die "API installation failed. Check logs: $LOG_FILE"
  fi
  
  # Write .env file (doing this separately to handle special characters)
  msg_info "Configuring API..."
  
  local env_content="# Paperless Configuration
PAPERLESS_URL=${PAPERLESS_URL}
PAPERLESS_TOKEN=${PAPERLESS_TOKEN}

# Ollama Configuration  
OLLAMA_URL=http://${OLLAMA_IP}:11434
OLLAMA_MODEL=${var_model}
OLLAMA_THREADS=10

# Processing Options
MAX_PAGES=3
AUTO_COMMIT=true
GENERATE_EXPLANATIONS=false
POLL_INTERVAL=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001"
  
  echo "$env_content" | pct exec "$API_CTID" -- tee /opt/paperless-classifier/.env >/dev/null
  
  # Start service
  pct exec "$API_CTID" -- systemctl enable paperless-classifier >/dev/null 2>&1
  pct exec "$API_CTID" -- systemctl start paperless-classifier
  
  msg_ok "Classifier API installed"
  
  # Wait for API
  msg_info "Waiting for API to start..."
  if wait_for_service "http://${API_IP}:8001/api/health" 30; then
    msg_ok "API responding at http://${API_IP}:8001"
  else
    msg_warn "API not responding yet (service may still be starting)"
    echo -e "${TAB}${YW}Check status: pct exec $API_CTID -- systemctl status paperless-classifier${CL}"
  fi
  
  log "API setup complete"
}

# =============================================================================
# CONFIRMATION DIALOG
# =============================================================================

show_summary() {
  local summary="
Paperless:
  URL:   $PAPERLESS_URL
  Token: ********

Ollama LXC (CT $OLLAMA_CTID):
  Hostname: $var_ollama_hostname
  CPU:      $var_ollama_cpu cores
  RAM:      ${var_ollama_ram}MB ($((var_ollama_ram/1024))GB)
  Disk:     ${var_ollama_disk}GB
  Model:    $var_model

Classifier API LXC (CT $API_CTID):
  Hostname: $var_api_hostname
  CPU:      $var_api_cpu cores
  RAM:      ${var_api_ram}MB
  Disk:     ${var_api_disk}GB

Network:
  Bridge:  $var_bridge
  VLAN:    ${var_vlan:-none}
  Storage: $CONTAINER_STORAGE

Proceed with installation?"

  if command -v whiptail &>/dev/null; then
    whiptail --backtitle "Proxmox VE Helper Scripts" \
      --title "INSTALLATION SUMMARY" \
      --yesno "$summary" 28 60
    return $?
  else
    echo -e "${BL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
    echo -e "${BL}              INSTALLATION SUMMARY                        ${CL}"
    echo -e "${BL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
    echo "$summary"
    read -rp "Proceed? [y/N]: " confirm
    [[ "$confirm" =~ ^[Yy]$ ]]
    return $?
  fi
}

# =============================================================================
# COMPLETION
# =============================================================================

show_completion() {
  SCRIPT_SUCCESS=true  # Mark as successful to prevent cleanup
  
  header_info
  
  echo -e "${GN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
  echo -e "${GN}              INSTALLATION COMPLETE!                        ${CL}"
  echo -e "${GN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${CL}"
  echo ""
  echo -e "${YW}Ollama LXC:${CL}"
  echo -e "${TAB}Container ID: ${BOLD}$OLLAMA_CTID${CL}"
  echo -e "${TAB}IP Address:   ${BOLD}$OLLAMA_IP${CL}"
  echo -e "${TAB}Ollama API:   http://$OLLAMA_IP:11434"
  echo -e "${TAB}Model:        $var_model"
  echo ""
  echo -e "${YW}Classifier API LXC:${CL}"
  echo -e "${TAB}Container ID: ${BOLD}$API_CTID${CL}"
  echo -e "${TAB}IP Address:   ${BOLD}$API_IP${CL}"
  echo -e "${TAB}Dashboard:    ${BGN}http://$API_IP:8001/dashboard${CL}"
  echo -e "${TAB}Config:       /opt/paperless-classifier/.env"
  echo ""
  echo -e "${YW}Paperless:${CL}"
  echo -e "${TAB}URL:          $PAPERLESS_URL"
  echo ""
  echo -e "${BL}Next Steps:${CL}"
  echo -e "${TAB}1. Open the dashboard: ${GN}http://$API_IP:8001/dashboard${CL}"
  echo -e "${TAB}2. Upload a document to Paperless"
  echo -e "${TAB}3. Watch it get automatically classified!"
  echo ""
  echo -e "${YW}Useful Commands:${CL}"
  echo -e "${TAB}View logs:     pct exec $API_CTID -- journalctl -u paperless-classifier -f"
  echo -e "${TAB}Edit config:   pct exec $API_CTID -- nano /opt/paperless-classifier/.env"
  echo -e "${TAB}Restart API:   pct exec $API_CTID -- systemctl restart paperless-classifier"
  echo -e "${TAB}Ollama shell:  pct enter $OLLAMA_CTID"
  echo ""
  echo -e "${DGN}Log file: $LOG_FILE${CL}"
  echo ""
  
  log "Installation completed successfully"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
  # Initialize log
  echo "=== Paperless AI Classifier Installer ===" > "$LOG_FILE"
  echo "Started: $(date)" >> "$LOG_FILE"
  echo "PID: $$" >> "$LOG_FILE"
  echo "" >> "$LOG_FILE"
  
  # Pre-flight checks
  preflight_checks
  
  # Paperless configuration
  configure_paperless
  
  # Storage selection
  header_info
  select_storage "vztmpl" "\nSelect storage for container templates:\n"
  TEMPLATE_STORAGE="$STORAGE_RESULT"
  msg_ok "Template storage: $TEMPLATE_STORAGE"
  
  select_storage "rootdir" "\nSelect storage for containers:\n"
  CONTAINER_STORAGE="$STORAGE_RESULT"
  msg_ok "Container storage: $CONTAINER_STORAGE"
  
  # Get container IDs
  OLLAMA_CTID=$(get_valid_container_id)
  if [[ -z "$OLLAMA_CTID" ]]; then
    die "Could not find available container ID"
  fi
  
  API_CTID=$(get_valid_container_id $((OLLAMA_CTID + 1)))
  if [[ -z "$API_CTID" ]]; then
    die "Could not find available container ID"
  fi
  
  log "Assigned CTIDs: Ollama=$OLLAMA_CTID, API=$API_CTID"
  
  # Advanced settings?
  local use_advanced=false
  if command -v whiptail &>/dev/null; then
    if whiptail --backtitle "Proxmox VE Helper Scripts" \
      --title "CONFIGURATION" \
      --yesno "Use advanced settings?\n\nAdvanced settings allow you to customize:\n• Vision model selection\n• CPU, RAM, and disk allocations\n• Network bridge and VLAN\n\nDefault settings work well for most users." 14 60; then
      use_advanced=true
    fi
  else
    read -rp "Use advanced settings? [y/N]: " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] && use_advanced=true
  fi
  
  [[ "$use_advanced" == "true" ]] && advanced_settings
  
  # Resource checks
  check_resources
  
  # Download template
  header_info
  download_template
  
  # Final confirmation
  if ! show_summary; then
    exit_script
  fi
  
  # === POINT OF NO RETURN ===
  # From here on, we're creating resources
  log "=== Starting container creation ==="
  
  header_info
  echo -e "${BL}Starting installation...${CL}\n"
  echo -e "${YW}Note: If interrupted, cleanup will be offered.${CL}\n"
  
  # Create and setup Ollama
  create_lxc "$OLLAMA_CTID" "$var_ollama_hostname" "$var_ollama_cpu" "$var_ollama_ram" "$var_ollama_disk" "Ollama"
  setup_ollama
  
  # Create and setup API
  create_lxc "$API_CTID" "$var_api_hostname" "$var_api_cpu" "$var_api_ram" "$var_api_disk" "API"
  setup_api
  
  # Done!
  show_completion
}

# Run main function
main "$@"
