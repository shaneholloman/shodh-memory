//! Graph Memory System - Inspired by Graphiti
//!
//! Temporal knowledge graph for tracking entities, relationships, and episodic memories.
//! Implements bi-temporal tracking and hybrid retrieval (semantic + graph traversal).

use anyhow::Result;
use chrono::{DateTime, Utc};
use rocksdb::{Options, DB};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

/// Entity node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Entity name (e.g., "John", "Paris", "Rust programming")
    pub name: String,

    /// Entity labels/types (e.g., ["Person"], ["Location", "City"])
    pub labels: Vec<EntityLabel>,

    /// When this entity was first created in the graph
    pub created_at: DateTime<Utc>,

    /// When this entity was last observed
    pub last_seen_at: DateTime<Utc>,

    /// How many times this entity has been mentioned
    pub mention_count: usize,

    /// Summary of this entity's context (built from surrounding edges)
    pub summary: String,

    /// Additional attributes based on entity type
    pub attributes: HashMap<String, String>,

    /// Semantic embedding of the entity name (for similarity search)
    pub name_embedding: Option<Vec<f32>>,

    /// Salience score (0.0 - 1.0): How important is this entity?
    /// Higher salience = larger gravitational well in the memory universe
    /// Factors: proper noun status, mention frequency, recency, user-defined importance
    #[serde(default = "default_salience")]
    pub salience: f32,

    /// Whether this is a proper noun (names, places, products)
    /// Proper nouns have higher base salience than common nouns
    #[serde(default)]
    pub is_proper_noun: bool,
}

fn default_salience() -> f32 {
    0.5 // Default middle salience
}

/// Entity labels following Graphiti's categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityLabel {
    Person,
    Organization,
    Location,
    Technology,
    Concept,
    Event,
    Date,
    Product,
    Skill,
    Other(String),
}

impl EntityLabel {
    /// Get string representation of the entity label
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::Person => "Person",
            Self::Organization => "Organization",
            Self::Location => "Location",
            Self::Technology => "Technology",
            Self::Concept => "Concept",
            Self::Event => "Event",
            Self::Date => "Date",
            Self::Product => "Product",
            Self::Skill => "Skill",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Relationship edge between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEdge {
    /// Unique identifier for this edge
    pub uuid: Uuid,

    /// Source entity UUID
    pub from_entity: Uuid,

    /// Target entity UUID
    pub to_entity: Uuid,

    /// Type of relationship
    pub relation_type: RelationType,

    /// Confidence/strength of this relationship (0.0 to 1.0)
    pub strength: f32,

    /// When this relationship was created
    pub created_at: DateTime<Utc>,

    /// When this relationship was last observed (temporal tracking)
    pub valid_at: DateTime<Utc>,

    /// Whether this relationship has been invalidated (temporal edge invalidation)
    pub invalidated_at: Option<DateTime<Utc>>,

    /// Source episode that created this relationship
    pub source_episode_id: Option<Uuid>,

    /// Additional context about the relationship
    pub context: String,
}

/// Relationship types following Graphiti's semantic model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationType {
    /// Work relationships
    WorksWith,
    WorksAt,
    EmployedBy,

    /// Structural relationships
    PartOf,
    Contains,
    OwnedBy,

    /// Location relationships
    LocatedIn,
    LocatedAt,

    /// Usage relationships
    Uses,
    CreatedBy,
    DevelopedBy,

    /// Causal relationships
    Causes,
    ResultsIn,

    /// Learning relationships
    Learned,
    Knows,
    Teaches,

    /// Generic relationships
    RelatedTo,
    AssociatedWith,

    /// Custom relationship
    Custom(String),
}

impl RelationType {
    /// Get string representation of the relation type
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::WorksWith => "WorksWith",
            Self::WorksAt => "WorksAt",
            Self::EmployedBy => "EmployedBy",
            Self::PartOf => "PartOf",
            Self::Contains => "Contains",
            Self::OwnedBy => "OwnedBy",
            Self::LocatedIn => "LocatedIn",
            Self::LocatedAt => "LocatedAt",
            Self::Uses => "Uses",
            Self::CreatedBy => "CreatedBy",
            Self::DevelopedBy => "DevelopedBy",
            Self::Causes => "Causes",
            Self::ResultsIn => "ResultsIn",
            Self::Learned => "Learned",
            Self::Knows => "Knows",
            Self::Teaches => "Teaches",
            Self::RelatedTo => "RelatedTo",
            Self::AssociatedWith => "AssociatedWith",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Episodic node representing a discrete experience/memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Human-readable name/title
    pub name: String,

    /// Episode content (the actual experience data)
    pub content: String,

    /// When the original event occurred (event time)
    pub valid_at: DateTime<Utc>,

    /// When this was ingested into the system (ingestion time)
    pub created_at: DateTime<Utc>,

    /// Entities extracted from this episode
    pub entity_refs: Vec<Uuid>,

    /// Source type
    pub source: EpisodeSource,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Episode source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpisodeSource {
    Message,
    Document,
    Event,
    Observation,
}

/// Graph memory storage and operations
pub struct GraphMemory {
    /// RocksDB storage for entities
    entities_db: Arc<DB>,

    /// RocksDB storage for relationships
    relationships_db: Arc<DB>,

    /// RocksDB storage for episodes
    episodes_db: Arc<DB>,

    /// RocksDB storage for entity -> relationships index
    entity_edges_db: Arc<DB>,

    /// RocksDB storage for entity -> episodes index (inverted index for fast lookup)
    entity_episodes_db: Arc<DB>,

    /// In-memory entity name index for fast lookups
    entity_name_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,
}

impl GraphMemory {
    /// Create a new graph memory system
    pub fn new(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let entities_db = Arc::new(DB::open(&opts, path.join("graph_entities"))?);
        let relationships_db = Arc::new(DB::open(&opts, path.join("graph_relationships"))?);
        let episodes_db = Arc::new(DB::open(&opts, path.join("graph_episodes"))?);
        let entity_edges_db = Arc::new(DB::open(&opts, path.join("graph_entity_edges"))?);
        let entity_episodes_db = Arc::new(DB::open(&opts, path.join("graph_entity_episodes"))?);

        // Rebuild entity name index from persisted data
        let entity_name_index = Self::rebuild_name_index(&entities_db)?;
        let index_count = entity_name_index.len();

        let graph = Self {
            entities_db,
            relationships_db,
            episodes_db,
            entity_edges_db,
            entity_episodes_db,
            entity_name_index: Arc::new(parking_lot::RwLock::new(entity_name_index)),
        };

        if index_count > 0 {
            tracing::info!("Rebuilt entity name index with {} entries", index_count);
        }

        Ok(graph)
    }

    /// Rebuild the entity name->UUID index from persisted database
    fn rebuild_name_index(entities_db: &DB) -> Result<HashMap<String, Uuid>> {
        let mut index = HashMap::new();

        let iter = entities_db.iterator(rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(entity) = bincode::deserialize::<EntityNode>(&value) {
                index.insert(entity.name.clone(), entity.uuid);
            }
        }

        Ok(index)
    }

    /// Add or update an entity node
    /// Salience is updated using the formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
    /// This means frequently mentioned entities grow in salience (gravitational wells get heavier)
    pub fn add_entity(&self, mut entity: EntityNode) -> Result<Uuid> {
        // Check if entity already exists by name
        let existing_uuid = {
            let index = self.entity_name_index.read();
            index.get(&entity.name).cloned()
        };

        if let Some(uuid) = existing_uuid {
            // Update existing entity
            entity.uuid = uuid;
            if let Some(existing) = self.get_entity(&uuid)? {
                entity.mention_count = existing.mention_count + 1;
                entity.last_seen_at = Utc::now();
                entity.created_at = existing.created_at; // Preserve original creation time
                entity.is_proper_noun = existing.is_proper_noun || entity.is_proper_noun;

                // Update salience with frequency boost
                // Formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
                // This caps at about 1.3x boost at 20 mentions
                let frequency_boost = 1.0 + 0.1 * (entity.mention_count as f32).ln();
                entity.salience = (existing.salience * frequency_boost).min(1.0);
            }
        } else {
            // New entity
            entity.uuid = Uuid::new_v4();
            entity.created_at = Utc::now();
            entity.last_seen_at = entity.created_at;
            entity.mention_count = 1;
            // Salience stays at base_salience for new entities
        }

        // Store in database
        let key = entity.uuid.as_bytes();
        let value = bincode::serialize(&entity)?;
        self.entities_db.put(key, value)?;

        // Update in-memory index
        {
            let mut index = self.entity_name_index.write();
            index.insert(entity.name.clone(), entity.uuid);
        }

        Ok(entity.uuid)
    }

    /// Get entity by UUID
    pub fn get_entity(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        let key = uuid.as_bytes();
        match self.entities_db.get(key)? {
            Some(value) => {
                let entity: EntityNode = bincode::deserialize(&value)?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Find entity by name
    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<EntityNode>> {
        let uuid = {
            let index = self.entity_name_index.read();
            index.get(name).cloned()
        };

        match uuid {
            Some(uuid) => self.get_entity(&uuid),
            None => Ok(None),
        }
    }

    /// Add a relationship edge
    pub fn add_relationship(&self, mut edge: RelationshipEdge) -> Result<Uuid> {
        edge.uuid = Uuid::new_v4();
        edge.created_at = Utc::now();

        // Store relationship
        let key = edge.uuid.as_bytes();
        let value = bincode::serialize(&edge)?;
        self.relationships_db.put(key, value)?;

        // Update entity->edges index for both entities
        self.index_entity_edge(&edge.from_entity, &edge.uuid)?;
        self.index_entity_edge(&edge.to_entity, &edge.uuid)?;

        Ok(edge.uuid)
    }

    /// Index an edge for an entity
    fn index_entity_edge(&self, entity_uuid: &Uuid, edge_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{edge_uuid}");
        self.entity_edges_db.put(key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Get all relationships for an entity
    pub fn get_entity_relationships(&self, entity_uuid: &Uuid) -> Result<Vec<RelationshipEdge>> {
        let mut edges = Vec::new();
        let prefix = format!("{entity_uuid}:");

        let iter = self.entity_edges_db.prefix_iterator(prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                // Extract edge UUID
                if let Some(edge_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(edge_uuid) = Uuid::parse_str(edge_uuid_str) {
                        if let Some(edge) = self.get_relationship(&edge_uuid)? {
                            edges.push(edge);
                        }
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Get relationship by UUID
    pub fn get_relationship(&self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.relationships_db.get(key)? {
            Some(value) => {
                let edge: RelationshipEdge = bincode::deserialize(&value)?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Add an episodic node
    pub fn add_episode(&self, episode: EpisodicNode) -> Result<Uuid> {
        let key = episode.uuid.as_bytes();
        let value = bincode::serialize(&episode)?;
        self.episodes_db.put(key, value)?;

        // Update inverted index: entity_uuid -> episode_uuid
        for entity_uuid in &episode.entity_refs {
            self.index_entity_episode(entity_uuid, &episode.uuid)?;
        }

        Ok(episode.uuid)
    }

    /// Index an episode for an entity (inverted index)
    fn index_entity_episode(&self, entity_uuid: &Uuid, episode_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{episode_uuid}");
        self.entity_episodes_db.put(key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Get episode by UUID
    pub fn get_episode(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
        let key = uuid.as_bytes();
        match self.episodes_db.get(key)? {
            Some(value) => {
                let episode: EpisodicNode = bincode::deserialize(&value)?;
                Ok(Some(episode))
            }
            None => Ok(None),
        }
    }

    /// Get all episodes that contain a specific entity
    ///
    /// Uses inverted index for O(k) lookup instead of O(n) full scan.
    /// Crucial for spreading activation algorithm.
    pub fn get_episodes_by_entity(&self, entity_uuid: &Uuid) -> Result<Vec<EpisodicNode>> {
        let mut episodes = Vec::new();
        let prefix = format!("{entity_uuid}:");

        // Use inverted index: entity_uuid -> episode_uuids
        let iter = self.entity_episodes_db.prefix_iterator(prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break; // Prefix iterator exhausted
                }

                // Extract episode UUID from key
                if let Some(episode_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(episode_uuid) = Uuid::parse_str(episode_uuid_str) {
                        if let Some(episode) = self.get_episode(&episode_uuid)? {
                            episodes.push(episode);
                        }
                    }
                }
            }
        }

        Ok(episodes)
    }

    /// Traverse graph starting from an entity (breadth-first)
    pub fn traverse_from_entity(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
    ) -> Result<GraphTraversal> {
        let mut visited_entities = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut current_level = vec![*start_uuid];
        let mut all_entities = Vec::new();
        let mut all_edges = Vec::new();

        visited_entities.insert(*start_uuid);
        if let Some(entity) = self.get_entity(start_uuid)? {
            all_entities.push(entity);
        }

        for _ in 0..max_depth {
            let mut next_level = Vec::new();

            for entity_uuid in current_level {
                let edges = self.get_entity_relationships(&entity_uuid)?;

                for edge in edges {
                    if visited_edges.contains(&edge.uuid) {
                        continue;
                    }

                    visited_edges.insert(edge.uuid);

                    // Only traverse non-invalidated edges
                    if edge.invalidated_at.is_some() {
                        continue;
                    }

                    all_edges.push(edge.clone());

                    // Add connected entity
                    let connected_uuid = if edge.from_entity == entity_uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    if !visited_entities.contains(&connected_uuid) {
                        visited_entities.insert(connected_uuid);
                        if let Some(entity) = self.get_entity(&connected_uuid)? {
                            all_entities.push(entity);
                        }
                        next_level.push(connected_uuid);
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            current_level = next_level;
        }

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Invalidate a relationship (temporal edge invalidation)
    pub fn invalidate_relationship(&self, edge_uuid: &Uuid) -> Result<()> {
        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            edge.invalidated_at = Some(Utc::now());

            let key = edge.uuid.as_bytes();
            let value = bincode::serialize(&edge)?;
            self.relationships_db.put(key, value)?;
        }

        Ok(())
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> Result<GraphStats> {
        let mut entity_count = 0;
        let mut relationship_count = 0;
        let mut episode_count = 0;

        let iter = self.entities_db.iterator(rocksdb::IteratorMode::Start);
        for _ in iter {
            entity_count += 1;
        }

        let iter = self.relationships_db.iterator(rocksdb::IteratorMode::Start);
        for _ in iter {
            relationship_count += 1;
        }

        let iter = self.episodes_db.iterator(rocksdb::IteratorMode::Start);
        for _ in iter {
            episode_count += 1;
        }

        Ok(GraphStats {
            entity_count,
            relationship_count,
            episode_count,
        })
    }

    /// Get all entities in the graph
    pub fn get_all_entities(&self) -> Result<Vec<EntityNode>> {
        let mut entities = Vec::new();

        let iter = self.entities_db.iterator(rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(entity) = bincode::deserialize::<EntityNode>(&value) {
                entities.push(entity);
            }
        }

        // Sort by mention count (most mentioned first)
        entities.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));

        Ok(entities)
    }
}

/// Result of graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversal {
    pub entities: Vec<EntityNode>,
    pub relationships: Vec<RelationshipEdge>,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub episode_count: usize,
}

/// Extracted entity with salience information
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub label: EntityLabel,
    pub is_proper_noun: bool,
    pub base_salience: f32,
}

/// Simple entity extraction (rule-based NER) with salience detection
pub struct EntityExtractor {
    /// Common person name indicators
    person_indicators: HashSet<String>,

    /// Common organization indicators
    org_indicators: HashSet<String>,

    /// Common technology keywords
    tech_keywords: HashSet<String>,

    /// Common words that should NOT be extracted as entities
    /// (stop words that start with capitals at sentence beginning)
    stop_words: HashSet<String>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let person_indicators: HashSet<String> =
            vec!["mr", "mrs", "ms", "dr", "prof", "sir", "madam"]
                .into_iter()
                .map(String::from)
                .collect();

        let org_indicators: HashSet<String> = vec![
            "inc",
            "corp",
            "ltd",
            "llc",
            "company",
            "corporation",
            "university",
            "institute",
            "foundation",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let tech_keywords: HashSet<String> = vec![
            "rust",
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "vue",
            "angular",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "sql",
            "nosql",
            "mongodb",
            "postgresql",
            "redis",
            "kafka",
            "api",
            "rest",
            "graphql",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Common words that are capitalized at sentence start but aren't entities
        let stop_words: HashSet<String> = vec![
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "i",
            "we",
            "you",
            "he",
            "she",
            "it",
            "they",
            "is",
            "are",
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "if",
            "when",
            "where",
            "what",
            "why",
            "how",
            "user",
            "error",
            "task",
            "code",
            "pattern",
            "search",
            "discovered",
            "completed",
            "performed",
            "mentioned",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            person_indicators,
            org_indicators,
            tech_keywords,
            stop_words,
        }
    }

    /// Calculate base salience for an entity based on its type and detection confidence
    fn calculate_base_salience(label: &EntityLabel, is_proper_noun: bool) -> f32 {
        let type_salience = match label {
            EntityLabel::Person => 0.8,       // People are highly salient
            EntityLabel::Organization => 0.7, // Organizations are important
            EntityLabel::Location => 0.6,     // Locations matter for context
            EntityLabel::Technology => 0.6,   // Tech keywords matter for dev context
            EntityLabel::Product => 0.7,      // Products are specific entities
            EntityLabel::Event => 0.6,        // Events are temporal anchors
            EntityLabel::Skill => 0.5,        // Skills are somewhat important
            EntityLabel::Concept => 0.4,      // Concepts are more generic
            EntityLabel::Date => 0.3,         // Dates are structural, not salient
            EntityLabel::Other(_) => 0.3,     // Unknown types get low salience
        };

        // Proper nouns get a 20% boost
        if is_proper_noun {
            (type_salience * 1.2_f32).min(1.0_f32)
        } else {
            type_salience
        }
    }

    /// Check if a word is likely a proper noun (not just capitalized at sentence start)
    fn is_likely_proper_noun(&self, word: &str, position: usize, prev_char: Option<char>) -> bool {
        // If it's not at position 0 and is capitalized, it's likely a proper noun
        if position > 0 {
            return true;
        }

        // At position 0, check if previous character was punctuation (sentence start)
        // If previous char was '.', '!', '?' then this might just be sentence capitalization
        if let Some(c) = prev_char {
            if c == '.' || c == '!' || c == '?' {
                // It's at sentence start - could be either
                // Check if it's a common word
                let lower = word.to_lowercase();
                return !self.stop_words.contains(&lower);
            }
        }

        // Default to proper noun for capitalized words
        true
    }

    /// Extract entities from text with salience information
    pub fn extract_with_salience(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();

        // Split into words and detect capitalized sequences
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());

            if clean_word.is_empty() || clean_word.len() < 2 {
                continue;
            }

            let lower = clean_word.to_lowercase();

            // Skip common stop words
            if self.stop_words.contains(&lower) {
                continue;
            }

            // Check for technology keywords (always proper nouns in tech context)
            if self.tech_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Technology,
                    is_proper_noun: true, // Tech keywords are always "proper" in context
                    base_salience: Self::calculate_base_salience(&EntityLabel::Technology, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for capitalized words (potential entities)
            if clean_word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                let mut entity_name = clean_word.to_string();
                let mut entity_label = EntityLabel::Other("Unknown".to_string());

                // Determine previous character for proper noun detection
                let prev_char = if i > 0 {
                    words[i - 1].chars().last()
                } else {
                    None
                };

                let is_proper = self.is_likely_proper_noun(clean_word, i, prev_char);

                // Check for person indicators
                if i > 0
                    && self
                        .person_indicators
                        .contains(&words[i - 1].to_lowercase())
                {
                    entity_label = EntityLabel::Person;
                }

                // Check for multi-word capitalized sequences
                let mut j = i + 1;
                while j < words.len()
                    && words[j]
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    let next_word = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                    // Skip stop words in multi-word sequences
                    if !self.stop_words.contains(&next_word.to_lowercase()) {
                        entity_name.push(' ');
                        entity_name.push_str(next_word);
                    }
                    j += 1;
                }

                // Check for organization indicators
                for word in entity_name.split_whitespace() {
                    if self.org_indicators.contains(&word.to_lowercase()) {
                        entity_label = EntityLabel::Organization;
                        break;
                    }
                }

                // Default to Person for single capitalized words without other indicators
                if matches!(entity_label, EntityLabel::Other(_)) && !entity_name.contains(' ') {
                    entity_label = EntityLabel::Person;
                } else if matches!(entity_label, EntityLabel::Other(_)) {
                    entity_label = EntityLabel::Concept;
                }

                let entity_key = entity_name.to_lowercase();
                if !seen.contains(&entity_key) {
                    let base_salience = Self::calculate_base_salience(&entity_label, is_proper);
                    let entity = ExtractedEntity {
                        name: entity_name,
                        label: entity_label,
                        is_proper_noun: is_proper,
                        base_salience,
                    };
                    entities.push(entity);
                    seen.insert(entity_key);
                }
            }
        }

        entities
    }

    /// Legacy method for backward compatibility - returns (name, label) tuples
    pub fn extract(&self, text: &str) -> Vec<(String, EntityLabel)> {
        self.extract_with_salience(text)
            .into_iter()
            .map(|e| (e.name, e.label))
            .collect()
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}
