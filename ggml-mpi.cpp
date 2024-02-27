#include "ggml-mpi.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

struct ggml_mpi_context {
    int rank;
    int size;
    MPI_Comm comm;
    int layer_start;
    int layer_end;
    struct ggml_tensor *inp0;
    std::string name;
    struct ggml_backend * wrapped_backend;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t scheduler;
};

void ggml_mpi_backend_init(void) {
    int ret;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &ret);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    auto * ctx = new ggml_mpi_context;

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);
    ctx->comm = MPI_COMM_WORLD;

    return ctx;
}

struct ggml_mpi_context * ggml_mpi_split_comm(struct ggml_mpi_context * ctx, int color, int key) {
    auto * newCtx = static_cast<ggml_mpi_context *>(calloc(1, sizeof(struct ggml_mpi_context)));
    MPI_Comm_split(ctx->comm, color, key, &newCtx->comm);
    MPI_Comm_rank(newCtx->comm, &newCtx->rank);
    MPI_Comm_size(newCtx->comm, &newCtx->size);
    return newCtx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    MPI_Comm_free(&(ctx->comm));
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

size_t ggml_mpi_size(struct ggml_mpi_context * ctx) {
    return ctx->size;
}

void ggml_mpi_eval_init(
        struct ggml_mpi_context *   ctx_mpi,
                int32_t         *   n_tokens,
                int32_t         **  pos,
                int32_t         **  n_seq_ids,
                int32_t         *** seq_id,
                int8_t          **  logits) {


    MPI_Barrier(ctx_mpi->comm);
    int32_t old_n_tokens = *n_tokens;
    MPI_Bcast(n_tokens, 1, MPI_INT32_T, 0, ctx_mpi->comm);

    // If what was passed in differs from what was broadcast,
    // we can't guarantee the allocated sizes are correct
    // TODO check how often this is done and if it's a problem,
    //      try to allocate ahead of time
    if (old_n_tokens != *n_tokens) {
        *pos = static_cast<int32_t *>(realloc(*pos, *n_tokens * sizeof(int32_t)));
        *n_seq_ids = static_cast<int32_t *>(realloc(*n_seq_ids, *n_tokens * sizeof(int32_t)));
        *logits = static_cast<int8_t *>(realloc(*logits, *n_tokens * sizeof(int32_t)));
    }



//    MPI_Bcast(&total_n_seq_ids,     1, MPI_INT32_T, 0, ctx_mpi->comm);
    MPI_Bcast(*n_seq_ids,   *n_tokens, MPI_INT32_T, 0, ctx_mpi->comm);

    // We need to know the total number of sequence
    // ids, so we count them all up
    int32_t total_n_seq_ids = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        total_n_seq_ids += (*n_seq_ids)[i];
    }

    // MPI can't chase the pointers for multidimensional arrays, so we flatten them first
    // for transit
    auto * flattened_seq_ids = static_cast<int32_t *>(calloc(total_n_seq_ids, sizeof(int32_t)));

    int32_t current_index = 0;

    // Only rank 0 needs to flatten since the others don't have the real seq_id
    if (ctx_mpi->rank == 0) {
        for (int32_t i = 0; i < *n_tokens; i++) {
            for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
                flattened_seq_ids[current_index] = (*seq_id)[i][j];
                current_index++;
            }
        }
    }


    MPI_Bcast(             *pos, *n_tokens,        MPI_INT32_T, 0, ctx_mpi->comm);
    MPI_Bcast(flattened_seq_ids,  total_n_seq_ids, MPI_INT32_T, 0, ctx_mpi->comm);
    //MPI_Bcast(*logits,               *n_tokens,        MPI_INT8_T, 0, ctx_mpi->comm);
    auto ** new_seq_id = static_cast<int32_t **>(calloc(*n_tokens, sizeof(int32_t *)));
    current_index = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        new_seq_id[i] = static_cast<int32_t *>(calloc((*n_seq_ids)[i], sizeof(int32_t)));
        for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
            new_seq_id[i][j] = flattened_seq_ids[current_index];
            current_index++;
        }
    }
    free(flattened_seq_ids);
    //free(*seq_id); // <- something is still holding onto this, need to investigate
    *seq_id = new_seq_id;
}

void ggml_mpi_synch_int(
        struct ggml_mpi_context * ctx_mpi,
                        int32_t * val
) {
    MPI_Bcast(val, 1, MPI_INT32_T, 0, ctx_mpi->comm);
}

static int ggml_graph_get_node_idx(struct ggml_cgraph * gf, const char * name) {
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == NULL) {
        fprintf(stderr, "%s: tensor %s not found\n", __func__, name);
        return -1;
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }

    fprintf(stderr, "%s: tensor %s not found in graph (should not happen)\n", __func__, name);
    return -1;
}


static void ggml_mpi_tensor_send(struct ggml_tensor * t, int mpi_rank_dst, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    const int retval = MPI_Send(t->data, ggml_nelements(t), mpi_type, mpi_rank_dst, 0, comm);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

static void ggml_mpi_tensor_recv(struct ggml_tensor * t, int mpi_rank_src, MPI_Comm comm) {
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    MPI_Status status; UNUSED(status);
    fprintf(stderr, "%s: tensor receive == null: %d\n", __func__, t->data == NULL);
    const int retval = MPI_Recv(t->data, ggml_nelements(t), mpi_type, mpi_rank_src, MPI_ANY_TAG, comm, &status);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    float node_weights[]
) {
    // Splits the range given by start and end
    // over the available nodes. This implementation
    // assumes that node 0 handles the final part of the range
    // while node 1 handles the beginning, to form a ring pipeline

    // Only node 0 deals with the device splits, other nodes
    // get the splits from the scatter layers operation

    if (ctx_mpi->rank != 0) {
        return NULL;
    }

    uint16_t range_length = end - start + 1;
    uint16_t ** ranges = (uint16_t**) malloc(sizeof(uint16_t*) * ctx_mpi->size);
    for (int i = 0; i < ctx_mpi->size; i++) {
        ranges[i] = (uint16_t*) malloc(sizeof(uint16_t) * 2);
    }
    uint16_t next_layer = 0;
    for (int i=1; i < ctx_mpi->size; i++) {
        ranges[i][0] = next_layer;
        ranges[i][1] = MIN(end, ranges[i][0] + (node_weights[i] * range_length) + start);
        next_layer = ranges[i][1];
    }

    ranges[0][0] = next_layer;
    ranges[0][1] = MIN(end, next_layer + (node_weights[0] * range_length) + start);
    return ranges;

}

void ggml_mpi_scatter_layers(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t ** layer_ranges
) {
    // Layer ranges is a 2d array with the first dimension
    // having a length of the number of nodes and the second
    // dimension having a length of 2. The inner arrays contain
    // the start and end layer ID for a node.
    uint16_t flattened_ranges[ctx_mpi->size * 2];

    if (layer_ranges != NULL) {
        for (int i = 0; i < ctx_mpi->size * 2; i += 2) {
            flattened_ranges[i] = layer_ranges[i/2][0];
            flattened_ranges[i + 1] = layer_ranges[i/2][1];
        }
    }

    uint16_t received_range[2];
    MPI_Scatter(flattened_ranges, 2, MPI_UINT16_T, received_range, 2, MPI_UINT16_T, 0, ctx_mpi->comm);
    ctx_mpi->layer_start = received_range[0];
    ctx_mpi->layer_end = received_range[1];
    fprintf(stderr, "Ranges for rank %d: [%d, %d]\n", ctx_mpi->rank, ctx_mpi->layer_start, ctx_mpi->layer_end);
}

void ggml_mpi_graph_creation_post(struct ggml_mpi_context * ctx_mpi, struct ggml_cgraph * gf, int   n_layers) {

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    ctx_mpi->inp0 = inp0;

//    fprintf(stderr, "gf->nodes[0] == %s\n", ggml_get_name(gf->nodes[0]));
//
//    GGML_ASSERT(inp0 == gf->nodes[0]);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input tokens to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //


    for (int i = 0; i < gf->n_nodes; i++) {
        gf->nodes[i]->backend = GGML_BACKEND_MPI_SPLIT;
    }


}

// TODO: there are many improvements that can be done to this implementation
void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = gf->nodes[0];
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = gf->nodes[0];
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    if (mpi_rank > 0) {
        if (mpi_rank == 1) {
            // the first node (1) receives the input tokens from the main node (0)
            if (inp_tokens->data == NULL) {

            }
            ggml_mpi_tensor_recv(inp_tokens, 0, ctx_mpi->comm);
        } else {
            // recv input data for each node into the "inp0" tensor (i.e. the first node in the compute graph)
            fprintf(stderr, "%s:%d: receiving layer inp0\n", __func__, ctx_mpi->rank);
            ggml_mpi_tensor_recv(inp0, mpi_rank - 1, ctx_mpi->comm);
        }
    } else if (mpi_size > 1) {
        // node 0 sends the input tokens to node 1
        ggml_mpi_tensor_send(inp_tokens, 1, ctx_mpi->comm);

        // recv the output data from the last node
        ggml_mpi_tensor_recv(inp0, mpi_size - 1, ctx_mpi->comm);
    }
}

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf) {

    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    // send the output data to the next node
    if (mpi_rank > 0) {
        ggml_mpi_tensor_send(gf->nodes[gf->n_nodes - 1], (mpi_rank + 1) % mpi_size, ctx_mpi->comm);
    }
}

// BACKEND V2

struct ggml_backend_mpi_buffer_type_context {
    std::string name;
    ggml_backend_buffer_type_t wrapped_buffer;
};

GGML_CALL static const char * ggml_backend_mpi_buffer_type_name(ggml_backend_buffer_type_t buft);

GGML_CALL static bool ggml_backend_mpi_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {

    struct ggml_mpi_context * ctx = (ggml_mpi_context *) backend->context;

    ggml_mpi_graph_compute_pre(ctx, cgraph);

    std::vector<ggml_backend_buffer_type_t> backend_buft;
    for (auto *curr_backend : ctx->backends) {
        if (ggml_backend_is_cpu(curr_backend)) {
            // use host buffers for the CPU backend compute buffer
            backend_buft.push_back(ggml_backend_cpu_buffer_type());
        } else {
            backend_buft.push_back(ggml_backend_get_default_buffer_type(curr_backend));
        }
    }

//    ggml_backend_t wrapped_backend = ctx->wrapped_backend;
//    bool ret = ggml_backend_graph_compute(wrapped_backend, cgraph);
    printf("Running MPI backend\n");

    std::vector<std::pair<ggml_backend_buffer_type_t, std::vector<ggml_backend_buffer_type_t>> > old_buffs(cgraph->n_nodes);
    std::vector<ggml_backend_buffer_type_t> old_view_buffs(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        old_buffs.push_back({cgraph->nodes[i]->buffer->buft,{}});
        if (cgraph->nodes[i]->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
            cgraph->nodes[i]->buffer->buft = ((ggml_backend_mpi_buffer_type_context *) cgraph->nodes[i]->buffer->buft->context)->wrapped_buffer;
            printf("Unwrapped buffer: %s\n", cgraph->nodes[i]->buffer->buft->iface.get_name(cgraph->nodes[i]->buffer->buft));
        }

        for (auto & src : cgraph->nodes[i]->src) {
            if (src == nullptr) {
                break;
            }
            old_buffs[i].second.push_back(src->buffer->buft);
            if (src->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
                src->buffer->buft = ((ggml_backend_mpi_buffer_type_context *) src->buffer->buft->context)->wrapped_buffer;
                printf("Unwrapped buffer src: %s\n", src->buffer->buft->iface.get_name(src->buffer->buft));
            }
        }

        auto *src = cgraph->nodes[i]->view_src;
        if(src != nullptr && src->buffer->buft != nullptr){
            old_view_buffs[i] = src->buffer->buft;
            if (src->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
                src->buffer->buft = ((ggml_backend_mpi_buffer_type_context *) src->buffer->buft->context)->wrapped_buffer;
                printf("Unwrapped view buffer src: %s\n", src->buffer->buft->iface.get_name(src->buffer->buft));
            }
        }
    }


    std::vector<ggml_backend_buffer_type_t > old_buffs_leaves;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        old_buffs_leaves.push_back(cgraph->leafs[i]->buffer->buft);
        if (cgraph->leafs[i]->buffer->buft->iface.get_name == ggml_backend_mpi_buffer_type_name) {
            cgraph->leafs[i]->buffer->buft = ((ggml_backend_mpi_buffer_type_context *) cgraph->leafs[i]->buffer->buft->context)->wrapped_buffer;
            printf("Unwrapped buffer: %s\n", cgraph->leafs[i]->buffer->buft->iface.get_name(cgraph->leafs[i]->buffer->buft));
        }
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(ctx->backends.data(), backend_buft.data(), ctx->backends.size(), cgraph->n_nodes);


    printf("Created new scheduler\n");
    ggml_backend_sched_init_measure(sched, cgraph);
    printf("Beginning sched graph compute\n");
    ggml_backend_sched_graph_compute(sched, cgraph);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        cgraph->nodes[i]->buffer->buft = old_buffs[i].first;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (cgraph->nodes[i]->src[j] == nullptr) {
                break;
            }
            cgraph->nodes[i]->src[j]->buffer->buft = old_buffs[i].second[j];
        }
        if(cgraph->nodes[i]->view_src != nullptr && cgraph->nodes[i]->view_src->buffer->buft != nullptr) {
            cgraph->nodes[i]->view_src->buffer->buft = old_view_buffs[i];
        }

    }

    for (int i = 0; i < cgraph->n_leafs; i++) {
        cgraph->leafs[i]->buffer->buft = old_buffs_leaves[i];
    }


    ggml_mpi_graph_compute_post(ctx, cgraph);

    return true;
}


static const char * ggml_backend_mpi_name(ggml_backend_t backend) {
    return "MPI";
}

static void ggml_backend_mpi_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    delete ctx;


    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_mpi_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_mpi_context *>(backend->context);

    return ggml_backend_mpi_wrap_buffer(ctx->backends.back()->iface.get_default_buffer_type(ctx->backends.back()));
}

GGML_CALL static bool ggml_backend_mpi_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_CPY:
            return op->type != GGML_TYPE_IQ2_XXS && op->type != GGML_TYPE_IQ2_XS; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == ggml_internal_get_type_traits(op->src[0]->type).vec_dot_type;
        default:
            return true;
    }

    GGML_UNUSED(backend);
}



std::vector<ggml_mpi_device> ggml_mpi_available_devices_internal() {
    static bool has_init = false;
    if (!has_init) {
        ggml_mpi_backend_init();
        has_init = true;
    }
    std::vector<ggml_mpi_device> devices;
    int s;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    devices.resize(s);
    for (int i = 0; i < s; i++) {
        devices[i] = ggml_mpi_device{
                i,
                ggml_mpi_init(),
                ("MPI_COMM_WORLD:" + std::to_string(i)).c_str(),
                1
        };
    }
    return devices;
}



GGML_CALL bool ggml_backend_is_mpi(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_mpi_name;
}


GGML_CALL static const char * ggml_backend_mpi_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;

    return strdup(((ctx->name + ":") + std::string(ctx->wrapped_buffer->iface.get_name(ctx->wrapped_buffer))).c_str());
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_mpi_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    ggml_backend_buffer_t buf = ctx->wrapped_buffer->iface.alloc_buffer(ctx->wrapped_buffer, size);
    buf->buft = ggml_backend_mpi_wrap_buffer(buf->buft);
    return buf;
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    return ctx->wrapped_buffer->iface.get_alignment(ctx->wrapped_buffer);
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    return ctx->wrapped_buffer->iface.get_max_size(ctx->wrapped_buffer);
}

GGML_CALL static size_t ggml_backend_mpi_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;
    return ctx->wrapped_buffer->iface.get_alloc_size(ctx->wrapped_buffer, tensor);
}

GGML_CALL static bool ggml_backend_mpi_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return backend != nullptr && ggml_backend_is_mpi(backend);
}

GGML_CALL static bool ggml_backend_mpi_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_mpi_buffer_type_context *) buft->context;

    return ctx->wrapped_buffer->iface.is_host(ctx->wrapped_buffer);
}


static std::map<ggml_backend_buffer_type_t, ggml_backend_buffer_type_t> cached_wrappers;

static std::map<ggml_backend_t *, ggml_backend_t> cached_backends;


GGML_CALL ggml_backend_buffer_type_t ggml_backend_mpi_wrap_buffer(ggml_backend_buffer_type_t buft) {

    if (cached_wrappers.find(buft) != cached_wrappers.end()) {
        return cached_wrappers[buft];
    }

    ggml_backend_buffer_type_i ggml_backend_mpi_buffer_type_interface = {
            /* .get_name         = */ ggml_backend_mpi_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_mpi_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_mpi_buffer_type_get_alignment,
            /* .get_max_size     = */ (buft->iface.get_max_size != nullptr ) ? ggml_backend_mpi_buffer_type_get_max_size : nullptr,
            /* .get_alloc_size   = */ (buft->iface.get_alloc_size != nullptr ) ? ggml_backend_mpi_buffer_type_get_alloc_size : nullptr,
            /* .supports_backend = */ ggml_backend_mpi_buffer_type_supports_backend,
            /* .is_host          = */ (buft->iface.is_host != nullptr ) ? ggml_backend_mpi_buffer_type_is_host : nullptr,
    };

    auto* ggml_backend_wrapped_buffer_type = new ggml_backend_buffer_type{
            /* .iface    = */ ggml_backend_mpi_buffer_type_interface,
            /* .context  = */ new ggml_backend_mpi_buffer_type_context{"MPI",buft},
    };

    cached_wrappers[buft] = ggml_backend_wrapped_buffer_type;

    return ggml_backend_wrapped_buffer_type;
}

ggml_backend_t ggml_backend_mpi_init(ggml_backend_t * wrapped_backends, size_t num_backends) {

    if (cached_backends.find(wrapped_backends) != cached_backends.end()) {
        return cached_backends[wrapped_backends];
    }

    ggml_mpi_context * ctx = ggml_mpi_init();
    std::vector<ggml_backend_t> wrapped_backends_v;
    for (size_t i = 0; i < num_backends; i++) {
        wrapped_backends_v.push_back(wrapped_backends[i]);
    }
    ctx->backends = wrapped_backends_v;

    struct ggml_backend_i mpi_backend_i = {
            /* .get_name                = */ ggml_backend_mpi_name,
            /* .free                    = */ ggml_backend_mpi_free,
            /* .get_default_buffer_type = */ ggml_backend_mpi_get_default_buffer_type,
            /* .set_tensor_async        = */ NULL,
            /* .get_tensor_async        = */ NULL,
            /* .cpy_tensor_async        = */ NULL,
            /* .synchronize             = */ NULL,
            /* .graph_plan_create       = */ NULL,
            /* .graph_plan_free         = */ NULL,
            /* .graph_plan_compute      = */ NULL,
            /* .graph_compute           = */ ggml_backend_mpi_graph_compute,
            /* .supports_op             = */ ggml_backend_mpi_supports_op,
    };

    auto *mpi_backend = new ggml_backend {
            /* .interface = */ mpi_backend_i,
            /* .context   = */ ctx,
    };

    cached_backends[wrapped_backends] = mpi_backend;

    return mpi_backend;
}

static ggml_backend_t ggml_backend_reg_mpi_init(const char * params, void * user_data) {
    // TODO check what the parameters are for. Could use it to setup the MPI comms and routes?
    GGML_UNUSED(params);
    auto * v = new std::vector<ggml_backend_t>();
    v->push_back(ggml_backend_cpu_init());
    return ggml_backend_mpi_init(v->data(), 1);
}




extern "C" GGML_CALL int ggml_backend_mpi_reg_devices();

int ggml_backend_mpi_reg_devices() {
    auto devices = ggml_mpi_available_devices_internal();
    for (const auto & device : devices) {
        ggml_backend_register(
                device.name,
                ggml_backend_reg_mpi_init,
                ggml_backend_mpi_wrap_buffer(ggml_backend_cpu_buffer_type()),
                reinterpret_cast<void *>(intptr_t(device.index))
        );
    }
    return devices.size();
}




